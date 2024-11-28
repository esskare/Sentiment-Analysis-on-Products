import requests
import threading
import time

# Performance Test: Response Time for Bulk Prediction
def test_performance_bulk_prediction():
    file_path = "C:/Users/USER/OneDrive/ESTHER'S WORK/Sentiment-Analysis-main \
    /Data/Predictions.csv"
    files = {"file": open(file_path, "rb")}

    start_time = time.time()
    response = requests.post("http://127.0.0.1:5000/predict", files=files)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    assert response.status_code == 200, f"Expected 200, but got {response.status_code}"
    assert elapsed_time < 10, f"Test failed! Response time: {elapsed_time} seconds"
    print("Performance test passed.")

# Scalability Test: Handling Multiple Concurrent Requests
def send_request():
    data = {"text": "I Love Echo!"}
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    assert response.status_code == 200, f"Expected 200, but got {response.status_code}"

def test_scalability_concurrent_requests():
    num_requests = 50
    threads = []
    
    for _ in range(num_requests):
        thread = threading.Thread(target=send_request)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print("Scalability test passed.")

# Maintainability Test: Code Structure and Documentation Check

# Running the tests
if __name__ == "__main__":
    test_performance_bulk_prediction()
    test_scalability_concurrent_requests()
