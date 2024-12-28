import React, { useState, useEffect } from "react"
import {
  Button,
  Form,
  Container,
  Row,
  Col,
  Spinner,
  Alert
} from "react-bootstrap"
import { Camera, Link, Upload } from "lucide-react"
import "bootstrap/dist/css/bootstrap.min.css"

const App = () => {
  const [input, setInput] = useState("")
  const [inputType, setInputType] = useState("image")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState("")
  const [error, setError] = useState(null)
  const [overlapped, setOverlapped] = useState(false)
  const [streamUrl, setStreamUrl] = useState('');

  useEffect(() => {
    if (inputType === 'video' || inputType === 'url') {
      setStreamUrl(`http://127.0.0.1:5000/api/process?input=${input}&inputType=${inputType}`);
    }
  }, [inputType, input]);

  const handleInputChange = e => {
    const file = e.target.files?.[0]
    if (file) {
      setInput(file)
      setInputType(file.type.startsWith("image/") ? "image" : "video")
    }
  }

  const handleUrlChange = e => {
    setInput(e.target.value)
    setInputType("url")
  }

  const handleCheckBox = () => {
    setOverlapped((prevState) => !prevState);
  };

  const handleSubmit = async e => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append("input", input)
    formData.append("inputType", inputType)
    formData.append("overlapped", overlapped)

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
      console.log(data)
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
            <Form.Group controlId="overlapCheckbox">
              <Form.Check
                type="checkbox"
                label="Overlapped Image"
                onChange={handleCheckBox}
                checked={overlapped}
              />
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
      <div>
          {inputType === 'video' || inputType === 'url' ? (
            <img src={streamUrl} alt="Video Stream" />
          ) : (
            <p>No video input to display</p>
          )}
      </div>
    </Container>
  )
}

export default App;
