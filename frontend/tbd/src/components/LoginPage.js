import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';

function LoginPage() {
  const navigate = useNavigate();

  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  const handleLoginSubmit = async (event) => {
    event.preventDefault();
    //const backendApiUrl = 'https://raiso-ieee-2025.onrender.com';
   // const backendApiUrl ='https://backend-ieee.onrender.com';
    const backendApiUrl = 'http://localhost:5001'; 
    try {
      const response = await fetch(`${backendApiUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: loginUsername, password: loginPassword })
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('authToken', data.token);
        localStorage.setItem('userAuthData', JSON.stringify(data.user));
        localStorage.setItem('profileData', JSON.stringify(data.profile));
        navigate('/home');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Login Failed');
      }
    } catch (error) {
      console.error('Login failed:', error);
      setLoginError(error.message || 'Login failed. Please try again.');
    }
  };

  return (
    <>
      <Navbar />
      <div className="container-fluid d-flex justify-content-center align-items-center py-5" style={{ backgroundColor: '#F4EDFF', minHeight: '90vh' }}>
        <div className="custom-form-box p-4 shadow-lg rounded" style={{ maxWidth: '500px', width: '100%', backgroundColor: 'white' }}>
          <div className="text-center mb-4">
            <h3>Welcome Back!</h3>
            <p>Please log in to continue.</p>
          </div>
          <form onSubmit={handleLoginSubmit}>
            <div className="mb-3">
              <label htmlFor="loginUsername" className="form-label">Email</label>
              <input
                type="text"
                className="form-control"
                id="loginUsername"
                value={loginUsername}
                onChange={(e) => setLoginUsername(e.target.value)}
                required
              />
            </div>
            <div className="mb-3">
              <label htmlFor="loginPassword" className="form-label">Password</label>
              <input
                type="password"
                className="form-control"
                id="loginPassword"
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
                required
              />
            </div>
            {loginError && <p className="text-danger mb-3">{loginError}</p>}
            <div className="d-grid">
              <button type="submit" className="btn custom-btn-post-color">Log In</button>
            </div>
          </form>
          <p className="mt-3 text-center">
            Don't have an account?{' '}
            <span className="text-primary" style={{ cursor: 'pointer' }} onClick={() => navigate('/onboarding')}>
              Sign Up here
            </span>
          </p>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default LoginPage;