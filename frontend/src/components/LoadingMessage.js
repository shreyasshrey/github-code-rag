import React from 'react'

const LoadingMessage = () => {
  return (
    <div className="message assistant loading">
      <div className="role">Assistant</div>

      <div className="loader">
        <div className="loader-dot"></div>
        <div className="loader-dot"></div>
        <div className="loader-dot"></div>
      </div>
    </div>
  )
}

export default LoadingMessage