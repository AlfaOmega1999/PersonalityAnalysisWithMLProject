import React from 'react';
import './App.css';
import Navbar from './components/Navbar';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages';
import Types from './pages/types';
import Predictor from './pages/predictor';
import Stats from './pages/stats';
import About from './pages/about';

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path='/' exact element={<Home/>} />
        <Route path='/types' element={<Types/>} />
        <Route path='/predictor' element={<Predictor/>} />
        <Route path='/stats' element={<Stats/>} />
        <Route path='/about' element={<About/>} />
      </Routes>
    </Router>
  );
}

export default App;