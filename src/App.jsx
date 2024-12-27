import React, { useState } from "react"
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
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

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

  const handleSubmit = async e => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append("input", input)
    formData.append("inputType", inputType)

    try {
      const response = await fetch("http://127.0.0.1:5000/api/process", {
        method: "POST",
        body: formData
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data.count)
    } catch (err) {
      setError(`An error occurred while processing the input: ${err.message}`)
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
      {result !== null && (
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

export default App
