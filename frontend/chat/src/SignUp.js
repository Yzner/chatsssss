import React, { useState } from 'react';
import './styles/Main.css'; 
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import logoImage from "./website/Pictures/logocs.png";

function Signup() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    firstname: '',
    lastname: '',
    email: '',
    password: ''
  });

  const handleChange = e => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async e => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:5000/signup', formData);
      alert('Registered successfully!');
      navigate('/login');
    } catch (err) {
      alert('Signup failed. Please try again.');
    }
  };

  return (
    <div className="auth-container">
      <form className="auth-form" onSubmit={handleSubmit}>
        <img src={logoImage} alt="Logo" className="logo-SIGN" />
        <h2>Register an account</h2>
        <div className="name-fields">
          <input type="text" name="firstname" placeholder="First name" value={formData.firstname} onChange={handleChange} required />
          <input type="text" name="lastname" placeholder="Last name" value={formData.lastname} onChange={handleChange} required />
        </div>
        <input type="email" name="email" placeholder="Email address" value={formData.email} onChange={handleChange} required />
        <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} required />
        
        <button type="submit" className="auth-button">Sign Up</button>
        <p className="terms">
          By signing up, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.
        </p>
        <hr />
        <p className="login-link">
          Already have an account? <button onClick={() => navigate('/login')}>Login</button>
        </p>
      </form>
    </div>
  );
}

export default Signup;
