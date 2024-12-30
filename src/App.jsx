import React, { useState, useEffect, useRef } from "react"
import {
  Button,
  Form,
  Container,
  Row,
  Col,
  Spinner,
  Alert,
  ProgressBar
} from "react-bootstrap"
import { Camera, Link, Upload } from "lucide-react"
import "bootstrap/dist/css/bootstrap.min.css"
import Papa from 'papaparse'; // Import PapaParse for CSV parsing
import './App.css';

const App = () => {
  const [input, setInput] = useState("")
  const [inputType, setInputType] = useState("image")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState("")
  const [error, setError] = useState(null)
  const [overlapped, setOverlapped] = useState(false)
  const [processTime, setProcessTime] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [selectedModel, setSelectedModel] = useState('ResNet50');
  const [groundTruth, setGroundTruth] = useState("Not available");
  const [csvData, setCsvData] = useState([]);

  const timerRef = useRef(null)
  const imgRef = useRef(null)

  useEffect(() => {
    // Load and parse the CSV data
    fetch('/src/labels_nonOL.csv')
      .then(response => response.text())
      .then(data => {
        const parsedData = Papa.parse(data, { header: true }).data;
        setCsvData(parsedData);
      })
      .catch(error => console.error('Error loading CSV data:', error));
  }, []);
  
  const handleInputChange = e => {
    const file = e.target.files?.[0]
    if (file) {
      setInput(file)
      setInputType(file.type.startsWith("image/") ? "image" : "video")
      
      // Extract the file name and search in the CSV data
      const fileName = file.name;
      const found = csvData.find(row => row.image_file_name === fileName);
      if (found) {
        setGroundTruth(found.count);
      } else {
        setGroundTruth("Not available");
      }
    }
  }
  
  const handleRadioChange = (event) => {
    setSelectedModel(event.target.value);
    setOverlapped(false);
  };

  const handleUrlChange = e => {
    setInput(e.target.value)
    setInputType("url")
  }

  const handleCheckBox = () => {
    setOverlapped((prevState) => !prevState);
    setSelectedModel(null);
  };

  const startTimer = () => {
    const startTime = Date.now()
    timerRef.current = setInterval(() => {
      setProcessTime(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)
  }

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const handleVideoProcessing = async (formData) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/api/process", {
        method: "POST",
        body: formData,
        mode: "cors",
        credentials: "include"
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Create a blob URL from the response
      const blob = new Blob([await response.blob()], { type: 'multipart/x-mixed-replace; boundary=frame' });
      const url = URL.createObjectURL(blob);

      if (imgRef.current) {
        imgRef.current.src = url;
      }

      return url;
    } catch (error) {
      throw error;
    }
  };

  const handleStopButton = () => {
    setIsProcessing(false);
    setLoading(false);
  }
  const handleOverlappedRadioChange = (event) => {
    setSelectedModel(event.target.value);
    setOverlapped(true);
  };

  const handleSubmit = async e => {
    e.preventDefault()
    setError(null)
    setResult(null)
    setProcessTime(0)

    setLoading(true)
    const formData = new FormData()
    formData.append("input", input)
    formData.append("inputType", inputType)
    formData.append("overlapped", overlapped)
    formData.append("selected_model", selectedModel)

    if (inputType === 'video' || inputType === 'url') {
      setLoading(true)
      setIsProcessing(true)
      startTimer()

      try {
        const streamUrl = await handleVideoProcessing(formData);
        
        // Create new image element for MJPEG stream
        if (imgRef.current) {
          imgRef.current.onload = () => setLoading(false);
          imgRef.current.onerror = (e) => {
            console.error("Image load error:", e);
            setError("Failed to load video stream");
            setLoading(false);
          };
        }
      } catch (err) {
        console.error("Error:", err)
        setError(`Failed to process video: ${err.message}`)
        setLoading(false)
        stopTimer()
      }
      return
    }

    if (inputType === 'image') {
      const imageUrl = URL.createObjectURL(input)
      if (imgRef.current) {
        imgRef.current.src = imageUrl
      }
    }

    setLoading(true)
    try {
      const response = await fetch("http://127.0.0.1:5000/api/process", {
        method: "POST",
        body: formData,
        mode: "cors", // Enable CORS
        credentials: "include" // Include credentials if needed
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}, ${response.error}`)
      }
      
      const contentType = response.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        throw new TypeError("Oops, we haven't got JSON!")
      }
      
      const data = await response.json()
      // console.log(data)
      if (data.error) {
        throw new Error(data.error)
      }
      if (Array.isArray(data) && Array.isArray(data[0])) {
        const [total, overlapped, original] = data[0].map(value => 
            Math.round(Number(value))
        );
        setResult(`Total: ${total}, Overlapped: ${overlapped}, Original: ${original}`);
      } else {
          setResult(`${data}`);
      }    
    } catch (err) {
      console.error("Error details:", err)
      setError(
        `An error occurred while processing the input: ${
          err instanceof Error ? err.message : String(err)
        }`
      )
    } finally {
      setLoading(false)
    }
  }


  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopTimer()
      if (imgRef.current?.src) {
        URL.revokeObjectURL(imgRef.current.src)
      }
    }
  }, [])

  return (
    <Container className="mt-5">
      <h1 className="text-center mb-4">People Counter</h1>
      <Form onSubmit={handleSubmit}>
        <Row className="mb-3">
          <Col>
            <Form.Group>
              <Form.Label>Upload Image/Video</Form.Label>
              <Form.Control
                type="file"
                onChange={handleInputChange}
                accept="image/*,video/*"
              />
            </Form.Group>
            <Form.Group controlId="overlapCheckbox" className="flex-container">
              <div>
                <Form.Check
                  type="radio"
                  label="Use VGG16 Model"
                  value="VGG16"
                  onChange={handleRadioChange}
                  checked={selectedModel === 'VGG16'}
                />
                <Form.Check
                  type="radio"
                  label="Use ResNet50 Model"
                  value="ResNet50"
                  onChange={handleRadioChange}
                  checked={selectedModel === 'ResNet50'}
                />
              </div>
              <div>
                <Form.Check
                  type="checkbox"
                  label="Overlapped Image"
                  onChange={handleCheckBox}
                  checked={overlapped}
                />
                <Form.Check
                  type="radio"
                  label="Use ResNet18 Model (Overlapping)"
                  value="ResNet18"
                  onChange={handleOverlappedRadioChange}
                  checked={selectedModel === 'ResNet18'}
                />
                <Form.Check
                  type="radio"
                  label="Use ResNet50 Model (Overlapping)"
                  value="ResNet50Overlapped"
                  onChange={handleOverlappedRadioChange}
                  checked={selectedModel === 'ResNet50Overlapped'}
                />
              </div>
          </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label>Or Enter URL</Form.Label>
              <Form.Control
                type="url"
                onChange={handleUrlChange}
                placeholder="https://example.com/webcam"
              />
            </Form.Group>
          </Col>
        </Row>
        <div className="d-grid">
          <Button variant="primary" type="submit" disabled={loading || !input}>
            {loading ? (
              <>
                <Spinner animation="border" size="sm" className="me-2" />
                Processing...
                <p variant="success">Loading {isProcessing ? "video" : (overlapped ? "ResNet18" : selectedModel)} ...</p>
              </>
            ) : (
              <>
                {inputType === "image" && <Camera className="me-2" />}
                {inputType === "video" && <Upload className="me-2" />}
                {inputType === "url" && <Link className="me-2" />}
                Process Input
              </>
            )}
          </Button>
        </div>
      </Form>
      {isProcessing && (
        <div className="mt-3">
          <p className="text-success">YoloV5 Model is running...</p>
          <ProgressBar animated now={100} />
          <p className="text-center mt-2">Processing Time: {processTime} seconds</p>
          <button type="button" className="btn btn-danger" onClick={handleStopButton}>Stop</button>
        </div>
      )}
      {(inputType === 'video' || inputType === 'url' || inputType === 'image') && (
        <div className="mt-3">
          <div className="ratio ratio-16x9">
            <img
              ref={imgRef}
              alt="Media Stream"
              className="w-50 h-50 object-fit-contain"
              style={{ backgroundColor: 'transparent' }}
            />
          </div>
          {inputType === 'image' && (
            <Alert variant="success" className="mt-3">
            Ground Truth: {groundTruth}
          </Alert>
          )}
        </div>
      )}
      {result !== "" && (
        <Alert variant="success" className="mt-3">
          Number of people detected: {result}
        </Alert>
      )}
      {error && (
        <Alert variant="danger" className="mt-3">
          {error}
        </Alert>
      )}
    </Container>
  )
}

export default App;
