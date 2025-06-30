import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import re
from urllib.parse import urlparse
import sqlite3
import os

# Import our prediction modules
from ml.predict import predict_url
from ml.deep_predict import predict_url_deep

@dataclass
class URLDetectionResult:
    url: str
    is_malicious: bool
    confidence: float
    model_used: str
    timestamp: float
    additional_info: Dict = None

class RealTimeURLDetector:
    def __init__(self, 
                 use_deep_learning=True, 
                 model_type='lstm',
                 cache_size=10000,
                 db_path='url_cache.db'):
        self.use_deep_learning = use_deep_learning
        self.model_type = model_type
        self.cache_size = cache_size
        self.db_path = db_path
        
        # Initialize caches
        self.memory_cache = {}
        self.url_queue = Queue()
        self.detection_results = Queue()
        
        # Setup database
        self.setup_database()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'malicious_detected': 0,
            'avg_processing_time': 0
        }
    
    def setup_database(self):
        """Setup SQLite database for URL caching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS url_cache (
                url TEXT PRIMARY KEY,
                is_malicious BOOLEAN,
                confidence REAL,
                model_used TEXT,
                timestamp REAL,
                additional_info TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON url_cache(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL for consistent caching"""
        url = url.lower().strip()
        # Remove trailing slash
        if url.endswith('/'):
            url = url[:-1]
        return url
    
    def get_cached_result(self, url: str) -> Optional[URLDetectionResult]:
        """Get cached result from memory or database"""
        normalized_url = self.normalize_url(url)
        
        # Check memory cache first
        if normalized_url in self.memory_cache:
            self.stats['cache_hits'] += 1
            return self.memory_cache[normalized_url]
        
        # Check database cache
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT is_malicious, confidence, model_used, timestamp, additional_info
            FROM url_cache WHERE url = ?
        ''', (normalized_url,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.stats['cache_hits'] += 1
            additional_info = json.loads(result[4]) if result[4] else {}
            detection_result = URLDetectionResult(
                url=normalized_url,
                is_malicious=bool(result[0]),
                confidence=result[1],
                model_used=result[2],
                timestamp=result[3],
                additional_info=additional_info
            )
            
            # Add to memory cache
            self.memory_cache[normalized_url] = detection_result
            return detection_result
        
        return None
    
    def cache_result(self, result: URLDetectionResult):
        """Cache detection result"""
        normalized_url = self.normalize_url(result.url)
        
        # Add to memory cache
        self.memory_cache[normalized_url] = result
        
        # Limit memory cache size
        if len(self.memory_cache) > self.cache_size:
            # Remove oldest entries
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k].timestamp)
            del self.memory_cache[oldest_key]
        
        # Add to database cache
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO url_cache 
            (url, is_malicious, confidence, model_used, timestamp, additional_info)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            normalized_url,
            result.is_malicious,
            result.confidence,
            result.model_used,
            result.timestamp,
            json.dumps(result.additional_info) if result.additional_info else None
        ))
        
        conn.commit()
        conn.close()
    
    def detect_url(self, url: str) -> URLDetectionResult:
        """Detect if URL is malicious using available models"""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.get_cached_result(url)
        if cached_result:
            return cached_result
        
        # Validate URL
        if not self.is_valid_url(url):
            return URLDetectionResult(
                url=url,
                is_malicious=False,
                confidence=0.0,
                model_used='invalid_url',
                timestamp=time.time(),
                additional_info={'error': 'Invalid URL format'}
            )
        
        try:
            # Use deep learning model if available
            if self.use_deep_learning:
                try:
                    pred, prob = predict_url_deep(url, self.model_type)
                    model_used = f'deep_{self.model_type}'
                except Exception as e:
                    self.logger.warning(f"Deep learning model failed: {e}")
                    # Fallback to traditional model
                    pred, prob = predict_url(url)
                    model_used = 'traditional'
            else:
                pred, prob = predict_url(url)
                model_used = 'traditional'
            
            result = URLDetectionResult(
                url=url,
                is_malicious=bool(pred),
                confidence=prob,
                model_used=model_used,
                timestamp=time.time(),
                additional_info={
                    'processing_time': time.time() - start_time,
                    'raw_prediction': pred,
                    'raw_probability': prob
                }
            )
            
            # Cache result
            self.cache_result(result)
            
            # Update stats
            self.stats['total_processed'] += 1
            if result.is_malicious:
                self.stats['malicious_detected'] += 1
            
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['total_processed'] - 1) + processing_time) 
                / self.stats['total_processed']
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting URL {url}: {e}")
            return URLDetectionResult(
                url=url,
                is_malicious=False,
                confidence=0.0,
                model_used='error',
                timestamp=time.time(),
                additional_info={'error': str(e)}
            )
    
    def process_url_queue(self):
        """Process URLs from queue in background thread"""
        while True:
            try:
                url = self.url_queue.get(timeout=1)
                if url is None:  # Shutdown signal
                    break
                
                result = self.detect_url(url)
                self.detection_results.put(result)
                
            except Exception as e:
                self.logger.error(f"Error in URL processing thread: {e}")
    
    def start_background_processing(self, num_workers=4):
        """Start background URL processing"""
        self.workers = []
        for _ in range(num_workers):
            worker = threading.Thread(target=self.process_url_queue)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop_background_processing(self):
        """Stop background URL processing"""
        for _ in self.workers:
            self.url_queue.put(None)  # Shutdown signal
        
        for worker in self.workers:
            worker.join()
    
    def add_url_for_processing(self, url: str):
        """Add URL to processing queue"""
        self.url_queue.put(url)
    
    def get_detection_result(self, timeout=1) -> Optional[URLDetectionResult]:
        """Get next detection result from queue"""
        try:
            return self.detection_results.get(timeout=timeout)
        except:
            return None
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear all caches"""
        self.memory_cache.clear()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM url_cache')
        conn.commit()
        conn.close()

class NetworkMonitor:
    """Monitor network traffic for URLs (simplified version)"""
    
    def __init__(self, detector: RealTimeURLDetector):
        self.detector = detector
        self.monitoring = False
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text using regex"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls
    
    def monitor_text_stream(self, text_stream):
        """Monitor text stream for URLs"""
        self.monitoring = True
        
        for text in text_stream:
            if not self.monitoring:
                break
            
            urls = self.extract_urls_from_text(text)
            for url in urls:
                self.detector.add_url_for_processing(url)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False

# Example usage and testing
def test_real_time_detector():
    """Test the real-time URL detector"""
    detector = RealTimeURLDetector(use_deep_learning=True, model_type='lstm')
    
    # Start background processing
    detector.start_background_processing(num_workers=2)
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://malicious-site.com/steal-data.php",
        "https://github.com",
        "http://suspicious-domain.net/popup.exe",
        "https://www.microsoft.com",
        "http://fake-login-page.com/login"
    ]
    
    print("Testing real-time URL detection...")
    
    # Add URLs to processing queue
    for url in test_urls:
        detector.add_url_for_processing(url)
        print(f"Added URL to queue: {url}")
    
    # Get results
    results = []
    for _ in range(len(test_urls)):
        result = detector.get_detection_result(timeout=5)
        if result:
            results.append(result)
            print(f"Result: {result.url} -> {'MALICIOUS' if result.is_malicious else 'SAFE'} "
                  f"(confidence: {result.confidence:.3f}, model: {result.model_used})")
    
    # Show stats
    stats = detector.get_stats()
    print(f"\nStats: {stats}")
    
    # Stop processing
    detector.stop_background_processing()

if __name__ == "__main__":
    test_real_time_detector() 