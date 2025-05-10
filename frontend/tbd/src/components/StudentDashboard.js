import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom'; // For navigation if needed
import { useEffect } from 'react';

function StudentDashboard() {
    // Mock data for interested posts - replace with actual data fetching
    //const profileData = JSON.parse(localStorage.getItem('profileData'));
    // const firstName = profileData.first_name
    // const lastName = profileData.last_name

   // const backendApiUrl = 'https://raiso-ieee-2025.onrender.com';
    //const backendApiUrl ='https://backend-ieee.onrender.com';

    const backendApiUrl = 'http://localhost:5001'; 
    const navigate = useNavigate();
    
    const authToken = localStorage.getItem('authToken');
    let userAuthData = null;
    let userId = null;
    const storedUserAuthData = localStorage.getItem('userAuthData');
    if (storedUserAuthData) 
        {
        try {
            userAuthData = JSON.parse(storedUserAuthData);
            userId = userAuthData?.id;
            } 
        catch (error) {
            console.error("Error parsing userAuthData:", error);
            }
        }
    console.log("Auth Token:", authToken);
    console.log("User ID:", userId);

    const [profile, setProfile] = useState(null);//get profile
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [allLikedJobs, setAllLikedJobs] = useState([]);

    useEffect(() => {
        setLoading(true);
        setError(null);

        const fetchLikedJobs = async () => {
            try
            {
                const res = await fetch(`${backendApiUrl}/interests/${userId}`)
                if (!res.ok){
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                const data = await res.json();
                setAllLikedJobs(data); //all my liked jobs
            }

            catch(error)
            {
                setError(error.message);//set error to be that message
                setAllLikedJobs([]);

            }
            finally 
            {
                setLoading(false);
            }
        };
        fetchLikedJobs();
            },
        [userId]);



    const [buttonText, setbuttonText] = useState("Apply");

    const handleDropdownItemClick = (event) => {
        const selectedText = event.target.textContent ;
        setbuttonText(selectedText);

    };

    const [expandedJobId, setExpandedJobId] = useState(null); // init all cards as non-expanded
    const toggleExpand = (id) => {
    setExpandedJobId(prevId => (prevId === id ? null : id));
    }

    const handleLogout = () => {
        localStorage.removeItem('authToken');
        localStorage.removeItem('userData');
        localStorage.removeItem('profileData');
        // Optionally clear other user-specific data
        // Optionally redirect to the login page
        navigate('/');
        };

    if (loading) 
        {
            return <p>Loading profile...</p>;
        }

    if (error) 
        {
            return <p>
                    Error loading profile. Try <span to="#" onClick={handleLogout}className="nav-link custom-dashboard text-decoration-none">logging in</span> again.
                </p>
        }

    if (!error || profile) {
    // const firstName = profile.first_name
    // const last_name = profile.last_name

    
    return (
        <div className="  d-flex full-viewport"> 
            <div className='row '>
                <div className="col-md-3 bg-fefae0 sidebar-color p-4 rounded shadow"> {/* Similar background to navbar, rounded corners, shadow */}

                    <div className="d-flex align-items-center mb-3 ">
                        <div className="me-2">
                            {/* User Icon (replace with your actual icon) */}
                            <i class="bi bi-person-fill fs-1 custom-icon-color"></i>
                        </div>
                    <div>
                    {/* <h5>{firstName} {lastName}</h5> */}
                    {/* Optional: User Role */}
                    <p className="text-muted small">Student</p>
                            </div>
                            </div>

                                {/* Separator Line */}
                    <hr className="border-secondary my-2" />


                    <h5 className="custom-posts-title mb-3">Dashboard Menu</h5>
                    <ul className="list-unstyled">
                        <li className="mb-2">
                            <Link to="/home" className="nav-link custom-dashboard text-decoration-none">Go to Postings</Link>
                        </li>
                        <li className="mb-2">
                            <Link to="/studentprofile" className="nav-link custom-dashboard text-decoration-none">Profile</Link>
                        </li>
                        <li className="mb-2">
                            <span to="#" onClick={handleLogout}className="nav-link custom-dashboard text-decoration-none">Logout</span>
                        </li>
                        {/* Add more sidebar links as needed */}
                    </ul>
                </div>

                {/* Main Content - List of Interested Posts */}
                <div className="col-md-9 p-4 ">
                    <h2 className="custom-posts-title mb-4">Interested Opportunities</h2>
                    {allLikedJobs.length > 0 ? (
                        allLikedJobs.map((job) => (
                <div className="card custom-card mt-3  custom-card-bg-color" key={job.id}>
                      <div className="card-header card-header-bg-color" >
                        {job.title}
                      </div>
                      <div className="card-body">
                        <div className='d-flex justify-content-between'>
                          <h5 className="card-title" style={{ cursor: 'pointer', textDecoration: 'underline' }} onClick={() => toggleExpand(job.id)} >{job.organization}</h5>
                          <h6>{job.location}</h6>
                        </div>
                        <p className="card-text">{job.description} </p>
                        {expandedJobId === job.id && (
                          <>
                          <p className="card-text mt-2"><b>Eligibility:</b> {job.requirement}</p>
                          <p className="card-text"><b>Skills:</b> {job.skills}</p>
                          </>
                        )} 
                        <div className='d-flex'>
                          <div className='col-9'>
                            <a href={job.url} className="btn btn-primary custom-btn-post-color" target="_blank"  rel="noopener noreferrer" style= {{cursor: 'pointer', backgroundColor:  '#B1DABE', color:  'black' }} >Applied</a>
                          </div>
                          <div className='col-3 d-flex justify-content-around'>
                            <i className='bi bi-hand-thumbs-up xl liked fs-3' ></i>
                            {/* <i className="bi bi-hand-thumbs-down"></i> */}
                          </div>
                        </div>
                      </div>
                    </div>

                        ))
                    ) : 
                    
                    
                    (
                        <p>You haven't shown interest in any opportunities yet.</p>
                    )}
                </div>
            </div>
            
        </div>
    );
}
}

export default StudentDashboard;



