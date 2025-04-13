

import React, { useState } from 'react';
import './styles/Main.css'; 
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function UserLoginScreen() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const handleChange = e => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:5000/login', formData);
      alert('Login successful!');
      // Save login state (if needed)
      localStorage.setItem('user', JSON.stringify(res.data));
      navigate('/home'); // or wherever your home/dashboard is
    } catch (err) {
      alert('Login failed. Please check your credentials.');
    }
  };

  return (
    <div className="auth-container">
      <form className="auth-form" onSubmit={handleSubmit}>
        <h2>Login to your account</h2>
        <input
          className="loginC"
          type="email"
          name="email"
          placeholder="Email address"
          value={formData.email}
          onChange={handleChange}
          required
        />
        <input
          className="loginC"
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
          required
        />

        <button type="submit" className="auth-button">Login</button>

        <hr />
        <p className="login-link">
          Don't have an account? <button onClick={() => navigate('/signup')}>Sign up</button>
        </p>
      </form>
    </div>
  );
}

export default UserLoginScreen;




