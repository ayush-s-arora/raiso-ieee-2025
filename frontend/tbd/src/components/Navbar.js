import React, { useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';

function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const isOn = (path) => location.pathname === path;

  const authToken = localStorage.getItem('authToken');

  useEffect(() => {
    // Redirect logged-in users away from login or onboarding
    if (authToken && (isOn('/login') || isOn('/onboarding'))) {
      navigate('/home');
    }
  }, [authToken, location.pathname, navigate]);

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userAuthData');
    localStorage.removeItem('profileData');
    navigate('/');
  };

  return (
    <nav className='navbar navbar-expand-lg custom-navbar-bg-color py-2'>
      <div className='container-fluid'>
        <Link className="navbar-brand" to="/">
          <img src="/logo512.png" alt="NUVolunteers Logo" style={{ height: '32px', marginRight: '10px' }} />
          NUVolunteers
        </Link>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse justify-content-between" id="navbarNav">
          <ul className="navbar-nav">
            {!isOn('/about') && (
              <li className="nav-item">
                <Link className="nav-link" to="/about">About Us</Link>
              </li>
            )}
          </ul>
          <ul className="navbar-nav align-items-center">
            {!authToken && !isOn('/onboarding') && (
              <li className="nav-item">
                <Link className="nav-link" to="/onboarding">Sign Up</Link>
              </li>
            )}
            {!authToken && !isOn('/login') && (
              <li className="nav-item">
                <Link className="nav-link" to="/login">Log In</Link>
              </li>
            )}
            {authToken && (
              <>
                <li className="nav-item">
                  <span className="nav-link" style={{ cursor: 'pointer' }} onClick={handleLogout}>Log Out</span>
                </li>
                 <li className="nav-item">
                  <Link className="nav-link" to="/profile">
                    <i className="bi bi-person-circle fs-5"></i>
                  </Link>
                </li>
              </>
            )}
          </ul>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;