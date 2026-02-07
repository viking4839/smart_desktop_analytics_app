import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App' // This looks for App.tsx in the same folder
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
)