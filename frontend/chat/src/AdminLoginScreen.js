import React, { useState } from 'react';
import axios from 'axios';
import './styles/loginadmin.css';

const AdminLoginScreen = () => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    department: '',
  });

  const toggleSignUp = () => setIsSignUp(!isSignUp);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isSignUp) {
      try {
        const response = await axios.post('http://localhost:5000/adminsignup', formData);
        alert(response.data.message);
      } catch (error) {
        console.error("Sign-up error:", error);
        alert("Sign-up request failed.");
      }
    } else {
      try {
        const response = await axios.post('http://localhost:5000/adminlogin', {
          email: formData.email,
          password: formData.password,
        });
        if (response.data.success) {
          localStorage.setItem('adminToken', response.data.token);
          window.location.href = '/adminscreen';
        } else {
          alert("Login failed: " + response.data.message);
        }
      } catch (error) {
        console.error("Login error:", error);
        alert("Login failed.");
      }
    }
  };

  return (
    <div>
      <h2>{isSignUp ? 'Admin Sign Up' : 'Admin Login'}</h2>
      <form onSubmit={handleSubmit}>
        {isSignUp && (
          <>
            <input
              type="text"
              name="firstName"
              placeholder="First Name"
              value={formData.firstName}
              onChange={handleChange}
              required
            />
            <input
              type="text"
              name="lastName"
              placeholder="Last Name"
              value={formData.lastName}
              onChange={handleChange}
              required
            />
            <select name="department" value={formData.department} onChange={handleChange} required>
              <option value="">Select Department</option>
              <option value="College of Science">College of Science</option>
              <option value="Engineering">Engineering</option>
              <option value="Other">Other</option>
            </select>
          </>
        )}
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formData.email}
          onChange={handleChange}
          required
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
          required
        />
        <button type="submit">{isSignUp ? 'Request Sign Up' : 'Login'}</button>
      </form>
      <button onClick={toggleSignUp}>
        {isSignUp ? 'Already have an account? Log in' : 'New admin? Request sign up'}
      </button>
    </div>
  );
};

export default AdminLoginScreen;
