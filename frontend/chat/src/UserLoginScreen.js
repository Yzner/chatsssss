import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './styles/Login.css';

const UserLoginScreen = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    role: 'guest',
  });
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setFormData({ ...formData, firstName: '', lastName: '', password: '', email: '', role: 'guest' });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  const handleAuth = async () => {
    const { email, password } = formData;
    try {
      const endpoint = isLogin ? '/login' : '/signup';
      const response = await axios.post(`http://localhost:5000${endpoint}`, formData);
  
      if (response.data.success) {
        localStorage.setItem('token', response.data.token); 
        navigate('/chatbotscreen'); 
      } else {
        setError(response.data.message || 'Authentication failed.');
      }
    } catch (err) {
      setError('Error connecting to the server. Please try again.');
    }
  };
  

  return (
    <div className="auth-container">
      <h2>{isLogin ? 'Login' : 'Sign Up'}</h2>
      {error && <p className="error">{error}</p>}
      {!isLogin && (
        <>
          <input
            type="text"
            name="firstName"
            placeholder="First Name"
            value={formData.firstName}
            onChange={handleChange}
          />
          <input
            type="text"
            name="lastName"
            placeholder="Last Name"
            value={formData.lastName}
            onChange={handleChange}
          />
        </>
      )}
      <input
        type="email"
        name="email"
        placeholder="Email"
        value={formData.email}
        onChange={handleChange}
      />
      <input
        type="password"
        name="password"
        placeholder="Password"
        value={formData.password}
        onChange={handleChange}
      />
      {!isLogin && (
        <select name="role" value={formData.role} onChange={handleChange}>
          <option value="guest">Guest</option>
          <option value="student">Student</option>
        </select>
      )}
      <button onClick={handleAuth}>{isLogin ? 'Login' : 'Sign Up'}</button>
      <p onClick={toggleAuthMode} className="toggle-auth">
        {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Log in'}
      </p>
    </div>
  );
};

export default UserLoginScreen;
