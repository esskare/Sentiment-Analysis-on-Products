import requests
def test_upload_csv_for_bulk_prediction():
    """
    Test the ability to upload a CSV file, process it,\
    and retrieve sentiment predictions for each entry.
    """
    # Arrange
    file_path = r"C:\Users\USER\OneDrive\ESTHER'S WORK\Sentiment-Analysis-main\Data\Predictions.csv"
    files = {'file': open(file_path, 'rb')}

    # Act: Upload the file to the system and trigger the prediction
    response = requests.post("http://127.0.0.1:5000/predict", files=files)
    
    # Assert: Check if the status is OK and if the result contains expected sentiment information
    assert response.status_code == 200, f"Expected 200 status, got {response.status_code}"
    assert "Predicted sentiment" in response.text, "Sentiment prediction not found in the response"
    print("Upload and prediction for CSV test passed.")

def test_single_text_prediction():
    """
    Test the ability to perform sentiment prediction on a single user-provided text input.
    """
    # Arrange
    data = {"text": "I really like the product!"}
    
    # Act: Send the request for sentiment prediction
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    
    # Assert: Check if the status is OK and if the result matches the expected sentiment
    assert response.status_code == 200, f"Expected 200 status, got {response.status_code}"
    assert "Positive" in response.text, "Prediction result was not 'Positive' as expected"
    print("Single text prediction test passed.")

def test_empty_or_invalid_input():
    """
    Test the ability to handle empty or invalid user input gracefully and return appropriate feedback.
    """
    # Arrange: Empty input text
    data = {"text": ""}
    
    # Act: Send the request for sentiment prediction with empty input
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    
    # Assert: Check if the system handles empty input and returns the correct error message or default result
    assert response.status_code == 200, f"Expected 200 status, got {response.status_code}"
    assert "Positive" in response.text, "Empty input should result in 'Negative' sentiment or an appropriate message"
    print("Empty input handling test passed.")

# Running the tests
if __name__ == "__main__":
    test_upload_csv_for_bulk_prediction()
    test_single_text_prediction()
    test_empty_or_invalid_input()
    print("All tests passed successfully.")
