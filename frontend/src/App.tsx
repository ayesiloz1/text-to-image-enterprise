import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [prompt, setPrompt] = useState('');
    const [image, setImage] = useState(null);

    const generateImage = async () => {
        const response = await axios.post('http://localhost:5000/generate', { prompt });
        setImage(response.data.output);
    };

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold">Text-to-Image Generator</h1>
            <input className="border p-2" type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            <button className="bg-blue-500 text-white p-2" onClick={generateImage}>Generate</button>
            {image && <img src={image} alt="Generated" className="mt-4"/>}
        </div>
    );
}

export default App;