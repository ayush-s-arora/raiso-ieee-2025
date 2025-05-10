import React, { useState } from 'react';
import Navbar from './Navbar';
import Footer from './Footer';




function Onboarding(){

    //state variables
    //const backendApiUrl = 'https://raiso-ieee-2025.onrender.com';
    const backendApiUrl = 'http://localhost:5001'; 
    //const backendApiUrl ='https://backend-ieee.onrender.com';


  const [registrationSuccess, setRegistrationSuccess] = useState(false);
  const [registrationError, setRegistrationError] = useState('');

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [retypePassword, setRetypePassword] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [city, setCity] = useState('');
  const [state, setState] = useState('');
  const [zip, setZip] = useState('');
  const [interests, setInterests] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (password !== retypePassword) {
      setRegistrationError('Passwords do not match');
      setRegistrationSuccess(false);
      return;
    }

    if (password.length < 6) {
      setRegistrationError('Password must be at least 6 characters long');
      setRegistrationSuccess(false);
      return;
    }

    const formData = {
      email,
      password,
      firstName,
      lastName,
      city,
      state,
      zipcode: zip,
      description: interests,
    };

    try {
      const res = await fetch(`${backendApiUrl}/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'User not added');
      }

      setRegistrationSuccess(true);
      setRegistrationError('');
    } catch (error) {
      setRegistrationError(error.message || 'Registration failed. Please try again.');
      setRegistrationSuccess(false);
    }
  };

  return (
    <>
    <Navbar></Navbar>
      <div className='container-fluid d-flex justify-content-center align-items-center py-5' style={{ backgroundColor: '#F4EDFF', minHeight: '90vh' }}>
        <div className='custom-form-box shadow-lg rounded' style={{ maxWidth: '700px', width: '100%', backgroundColor: 'white', padding: '2rem' }}>
          
          <div className='text-start mb-4'>
            <h2 className='mb-2'>Registration</h2>
            <p className='text-muted'>Create your account to get matched with personalized volunteer opportunities near you.</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className='text-start mb-3'>
              <h5 className='mb-1'>Basic Information</h5>
              <p className='text-muted'>Tell us a bit about yourself.</p>
            </div>

            <div className='mb-3'>
              <label htmlFor='inputEmail' className='form-label'>Email</label>
              <input type='email' className='form-control' id='inputEmail' placeholder='example@gmail.com' value={email} onChange={e => setEmail(e.target.value)} />
            </div>

            <div className='row'>
              <div className='col-md-6 mb-3'>
                <label htmlFor='inputPassword1' className='form-label'>Password</label>
                <input type='password' className='form-control' id='inputPassword1' value={password} onChange={e => setPassword(e.target.value)} />
              </div>
              <div className='col-md-6 mb-3'>
                <label htmlFor='inputPassword2' className='form-label'>Retype Password</label>
                <input type='password' className='form-control' id='inputPassword2' value={retypePassword} onChange={e => setRetypePassword(e.target.value)} />
              </div>
            </div>

            <div className='row'>
              <div className='col-md-6 mb-3'>
                <label htmlFor='firstname' className='form-label'>First Name</label>
                <input type='text' className='form-control' id='firstname' placeholder='Willie' value={firstName} onChange={e => setFirstName(e.target.value)} />
              </div>
              <div className='col-md-6 mb-3'>
                <label htmlFor='lastname' className='form-label'>Last Name</label>
                <input type='text' className='form-control' id='lastname' placeholder='Wildcat' value={lastName} onChange={e => setLastName(e.target.value)} />
              </div>
            </div>

            <div className='row'>
              <div className='col-md-6 mb-3'>
                <label htmlFor='inputCity' className='form-label'>City</label>
                <input type='text' className='form-control' id='inputCity' value={city} onChange={e => setCity(e.target.value)} />
              </div>
              <div className='col-md-4 mb-3'>
                <label htmlFor='inputState' className='form-label'>State</label>
                <select id='inputState' className='form-select' value={state} onChange={e => setState(e.target.value)}>
                  <option value="">Choose...</option>
                    <option value="AL">Alabama</option>
                    <option value="AK">Alaska</option>
                    <option value="AZ">Arizona</option>
                    <option value="AR">Arkansas</option>
                    <option value="CA">California</option>
                    <option value="CO">Colorado</option>
                    <option value="CT">Connecticut</option>
                    <option value="DE">Delaware</option>
                    <option value="FL">Florida</option>
                    <option value="GA">Georgia</option>
                    <option value="HI">Hawaii</option>
                    <option value="ID">Idaho</option>
                    <option value="IL">Illinois</option>
                    <option value="IN">Indiana</option>
                    <option value="IA">Iowa</option>
                    <option value="KS">Kansas</option>
                    <option value="KY">Kentucky</option>
                    <option value="LA">Louisiana</option>
                    <option value="ME">Maine</option>
                    <option value="MD">Maryland</option>
                    <option value="MA">Massachusetts</option>
                    <option value="MI">Michigan</option>
                    <option value="MN">Minnesota</option>
                    <option value="MS">Mississippi</option>
                    <option value="MO">Missouri</option>
                    <option value="MT">Montana</option>
                    <option value="NE">Nebraska</option>
                    <option value="NV">Nevada</option>
                    <option value="NH">New Hampshire</option>
                    <option value="NJ">New Jersey</option>
                    <option value="NM">New Mexico</option>
                    <option value="NY">New York</option>
                    <option value="NC">North Carolina</option>
                    <option value="ND">North Dakota</option>
                    <option value="OH">Ohio</option>
                    <option value="OK">Oklahoma</option>
                    <option value="OR">Oregon</option>
                    <option value="PA">Pennsylvania</option>
                    <option value="RI">Rhode Island</option>
                    <option value="SC">South Carolina</option>
                    <option value="SD">South Dakota</option>
                    <option value="TN">Tennessee</option>
                    <option value="TX">Texas</option>
                    <option value="UT">Utah</option>
                    <option value="VT">Vermont</option>
                    <option value="VA">Virginia</option>
                    <option value="WA">Washington</option>
                    <option value="WV">West Virginia</option>
                    <option value="WI">Wisconsin</option>
                    <option value="WY">Wyoming</option>
                </select>
              </div>
              <div className='col-md-2 mb-3'>
                <label htmlFor='inputZip' className='form-label'>Zip</label>
                <input type='text' className='form-control' id='inputZip' value={zip} onChange={e => setZip(e.target.value)} />
              </div>
            </div>

            <div className='text-start mb-3'>
              <h5 className='mb-1'>Skills & Interests</h5>
              <p className='text-muted'>Let us know what you’re passionate about—we’ll match you accordingly.</p>
            </div>

            <div className='mb-4'>
              <label htmlFor='interests' className='form-label'>Describe your interests</label>
              <textarea className='form-control' id='interests' rows='3' value={interests} onChange={e => setInterests(e.target.value)}></textarea>
            </div>

            <div className='text-center mt-3'>
              <button type='submit' className='btn custom-btn-post-color px-4 py-2'>Submit</button>
            </div>

            {registrationSuccess && (
              <div className='mt-3 alert alert-success' role='alert'>
                Registration successful! Check your email to confirm your address.
              </div>
            )}
            {registrationError && (
              <div className='mt-3 alert alert-danger' role='alert'>
                Error: {registrationError}
              </div>
            )}
          </form>
        </div>
      </div>
      <Footer />
    </>
  );
}

export default Onboarding;
