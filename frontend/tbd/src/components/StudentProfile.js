import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

<<<<<<< HEAD
function StudentProfile(){
    //get user data from local storage
    //const backendApiUrl = 'https://raiso-ieee-2025.onrender.com';
   // const backendApiUrl ='https://backend-ieee.onrender.com';

    const backendApiUrl = 'http://localhost:5001'; 
    const navigate = useNavigate();
=======
function StudentProfile() {
  const backendApiUrl = 'https://backend-ieee.onrender.com';
  const navigate = useNavigate();
>>>>>>> 7124eb12b7e9b8e37e10e732126e4be7292820ba

  const authToken = localStorage.getItem('authToken');
  let userAuthData = null;
  let userId = null;
  const storedUserAuthData = localStorage.getItem('userAuthData');
  if (storedUserAuthData) {
    try {
      userAuthData = JSON.parse(storedUserAuthData);
      userId = userAuthData?.id;
    } catch (error) {
      console.error("Error parsing userAuthData:", error);
    }
  }

  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [email, setEmail] = useState("");
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [city, setCity] = useState('');
  const [state, setState] = useState('');
  const [zip, setZip] = useState('');
  const [interests, setInterests] = useState('');

  useEffect(() => {
    if (!authToken) {
      navigate('/');
      return;
    }

    const getProfile = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${backendApiUrl}/auth/me`, {
          headers: {
            'Authorization': `Bearer ${authToken}`,
          },
        });
        if (!res.ok) {
          if (res.status === 401) {
            localStorage.removeItem('authToken');
            navigate('/');
            return;
          }
          const errorData = await res.json();
          throw new Error(errorData.error || 'Failed to fetch profile');
        }
        const data = await res.json();
        setProfile(data.user);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };
    getProfile();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = {
      email: email || profile?.[0]?.email,
      firstName: firstName || profile?.[0]?.first_name,
      lastName: lastName || profile?.[0]?.last_name,
      city: city || profile?.[0]?.city,
      state: state || profile?.[0]?.state,
      zipcode: zip || profile?.[0]?.zipcode,
      description: interests || profile?.[0]?.description,
    };

    try {
      const res = await fetch(`${backendApiUrl}/auth/updateme`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Failed to update profile');
      }

      const data = await res.json();
      console.log('Profile updated successfully:', data);
    } catch (error) {
      console.error('Update failed:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
    localStorage.removeItem('profileData');
    navigate('/');
  };

  if (loading) return <p>Loading profile...</p>;
  if (error) return <p>Error loading profile: {error}</p>;

  const user = profile?.[0];

  return (
    <div className='row m-0'>
      <div className="col-md-3 bg-fefae0 sidebar-color p-4 rounded shadow">
        <hr className="border-secondary my-2" />
        <h5 className="custom-posts-title mb-3">Dashboard Menu</h5>
        <ul className="list-unstyled">
          <li className="mb-2 ms-2">
            <Link to="/home" className="nav-link custom-dashboard text-decoration-none">Go to Postings</Link>
          </li>
          <li className="mb-2 ms-2">
            <Link to="/studentprofile" className="nav-link custom-dashboard text-decoration-none">Profile</Link>
          </li>
          <li className="mb-2 ms-2">
            <span role="button" onClick={handleLogout} className="nav-link custom-dashboard text-decoration-none">Logout</span>
          </li>
        </ul>
      </div>

      <div className='col-md-9 px-4 py-4'>
        <div className='custom-form-box'>
          <h3 className='text-center custom-landing-text mb-4'>Let's update your profile</h3>

          <form onSubmit={handleSubmit}>
            <div className='custom-form-section custom-form-blurb-color p-3 rounded mb-4'>
              <div className="mb-3">
                <label htmlFor="inputEmail" className="form-label">Email</label>
                <input type="email" className="form-control" id="inputEmail" defaultValue={user?.email} onChange={e => setEmail(e.target.value)} />
              </div>
              <div className='row g-3'>
                <div className="col-md-6">
                  <label htmlFor="firstname" className="form-label">First Name</label>
                  <input type="text" className="form-control" id="firstname" defaultValue={user?.first_name} onChange={e => setFirstName(e.target.value)} />
                </div>
                <div className="col-md-6">
                  <label htmlFor="lastname" className="form-label">Last Name</label>
                  <input type="text" className="form-control" id="lastname" defaultValue={user?.last_name} onChange={e => setLastName(e.target.value)} />
                </div>
              </div>
              <div className='row g-3 mt-3'>
                <div className="col-md-6">
                  <label htmlFor="inputCity" className="form-label">City</label>
                  <input type="text" className="form-control" id="inputCity" defaultValue={user?.city} onChange={e => setCity(e.target.value)} />
                </div>
                <div className="col-md-4">
                  <label htmlFor="inputState" className="form-label">State</label>
                  <select id="inputState" className="form-select" onChange={e => setState(e.target.value)} defaultValue={user?.state || 'Choose...'}>
                    <option disabled>Choose...</option>
                    {["AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"].map(state => (
                      <option key={state} value={state}>{state}</option>
                    ))}
                  </select>
                </div>
                <div className="col-md-2">
                  <label htmlFor="inputZip" className="form-label">Zip</label>
                  <input type="text" className="form-control" id="inputZip" defaultValue={user?.zipcode} onChange={e => setZip(e.target.value)} />
                </div>
              </div>
            </div>

            <div className='custom-form-section custom-form-blurb-color p-3 rounded mb-4'>
              <label htmlFor="interests" className="form-label">Skills and Interests</label>
              <textarea className='form-control' id="interests" rows='3' defaultValue={user?.description} onChange={e => setInterests(e.target.value)}></textarea>
              <div className="form-text mt-1">Tell us what you're passionate about. We'll use this to suggest great opportunities.</div>
            </div>

            <div className='text-end'>
              <button type='submit' className='btn btn-primary custom-btn-post-color rounded-pill shadow'>Save Changes</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default StudentProfile;