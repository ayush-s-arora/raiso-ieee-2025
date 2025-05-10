import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import  { useState, useEffect } from 'react';

function StudentProfile(){
    //get user data from local storage
    //const backendApiUrl = 'https://raiso-ieee-2025.onrender.com';
   // const backendApiUrl ='https://backend-ieee.onrender.com';

    const backendApiUrl = 'http://localhost:5001'; 
    const navigate = useNavigate();

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
    console.log("Auth Token:", authToken);
    console.log("User ID:", userId);

    const [profile, setProfile] = useState(null);//get profile
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [email, setEmail] = useState(""); //"" is the intial value 
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [city, setCity] = useState('');
    const [state, setState] = useState('');
    const [zip, setZip] = useState('');
    const [interests, setInterests] = useState('');
    


    useEffect(() => { 
        if (!authToken)//if there is no token
        {
            navigate('/');
            return;
        }

        const getProfile = async () => {

            setLoading(true);

            try
            {
                const res = await fetch(`${backendApiUrl}/auth/me`, {
                    headers: {
                        'Authorization' :  `Bearer ${authToken}`,
                    },
                });//fetch profile with the token to authorize

                if (!res.ok)
                {
                    if (res.status === 401)
                    {
                        localStorage.removeItem('authToken');//token expired
                        navigate('/');
                        return;
                    }
                    const errorData = await res.json();
                    throw new Error(errorData.error || 'Failed to fetch profile');
                }
                const data = await res.json();
                console.log('data on frontend', data);
                setProfile(data.user);

            }

            catch(error)
            {
                setError(error.message);

            }
            finally
            {
                setLoading(false);
            }
        };
        getProfile();
    }, []);

    console.log('profile', profile)

    const handleSubmit = async (e) => { 
        e.preventDefault();

        const formData = {
            email: email === "" && profile && profile[0]?.email ? profile[0].email : email,
            firstName: firstName === "" && profile && profile[0]?.first_name ? profile[0].first_name : firstName,
            lastName: lastName === "" && profile && profile[0]?.last_name ? profile[0].last_name : lastName,
            city: city === "" && profile && profile[0]?.city ? profile[0].city : city,
            state: state === "" && profile && profile[0]?.state ? profile[0].state : state,
            zipcode: zip === "" && profile && profile[0]?.zipcode ? profile[0].zipcode : zip,
            description: interests === "" && profile && profile[0]?.description ? profile[0].description : interests,
        };
        try
        {
            const res = await fetch(`${backendApiUrl}/auth/updateme`, { // Or your specific update profile endpoint
            method: 'POST', // Or 'PATCH' depending on your API design
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`, // Ensure you're sending the auth token
            },
            body: JSON.stringify(formData),
        });

        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.error || 'Failed to update profile');
        }

        const data = await res.json();
        console.log('Profile updated successfully:', data);
        // Optionally, provide user feedback (e.g., a success message)
        // You might also want to update the local 'profile' state with the new data
        //setProfile(data.user); 
        }
        catch(error)
        {
            console.error('Update failed:', error);
        }
    };


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
            return <p>Error loading profile: {error}</p>;
        }

    if (profile){
        const firstNameDisplay = profile[0].first_name;
        const lastNameDisplay = profile[0].last_name;
        const emailDisplay = profile[0]?.email;
        const cityDisplay = profile[0]?.city;
        const stateDisplay = profile[0]?.state;
        const zipDisplay = profile[0]?.zipcode;
        const descriptionDisplay = profile[0]?.description
        // setEmail(profile[0]?.email || '');
        // setFirstName(profile[0]?.first_name || '');
        // setLastName(profile[0]?.last_name || '');
        // setCity(profile[0]?.city || '');
        // setState(profile[0]?.state || '');
        // setZip(profile[0]?.zipcode || '');
    

    return(

        <div className=' row'>
            <div className="col-md-3 bg-fefae0 sidebar-color p-4 rounded shadow"> {/* Similar background to navbar, rounded corners, shadow */}

                    <div className="d-flex align-items-center mb-3">
                        <div className="me-2">
                            {/* User Icon (replace with your actual icon) */}
                            <i class="bi bi-person-fill fs-1 custom-icon-color"></i>
                        </div>
                    <div>
                    <h6>{firstNameDisplay} {lastNameDisplay}</h6>
                    {/* Optional: User Role */}
                    <p className="text-muted small">Student</p>
                            </div>
                            </div>

                                {/* Separator Line */}
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
                        {/* Add more sidebar links as needed */}
                    </ul>
                </div>



                <div className=' col-md-9 container-fluid d-flex justify-content-center align-items-center'>
                    <div className='custom-form-box d-flex flex-column align-items-center'>
                        
                        <div className='container-fluid text-center mt-4 custom-landing-text'>
                            <h3>Student Profile</h3>
                        </div>

                        <div className='container-fluid mt-4 ps-5 pe-5 pb-3'>
                            <form>
                                <div>
                                    <h5 className='custom-landing-text mb-0'>Personal Information</h5>
                                    <p className='ps-3'>Input your profile info here!</p>
                                </div>

                                <div className='container pt-1 custom-form-section custom-form-blurb-color'>
                                    <div class="row g-3 pb-3">
                                        <div class="col">
                                            <label for="inputEmail" class="form-label" >Email</label>
                                                <input type="email" class="form-control" placeholder='example@gmail.com' id="inputEmail"  defaultValue={emailDisplay} onChange={e => setEmail(e.target.value)}/>
                                        </div>
                                    </div>


                                    <div class='row g-3 pb-3'>
                                        <div class="col-md-6">
                                            <label for="firstname" class="form-label">First Name</label>
                                                <input type="text" class="form-control" placeholder='Willie' id="firstname"  defaultValue={firstNameDisplay} onChange={e => setFirstName(e.target.value)}/>
                                        </div>

                                        <div class="col-md-6">
                                            <label for="lastname" class="form-label">Last Name</label>
                                                <input type="text" class="form-control"placeholder='Wildcat' id="lastname"  defaultValue={lastNameDisplay} onChange={e => setLastName(e.target.value)}/>
                                        </div>

                                    </div>
                                    
                                    <div className='row g-3 pb-3'>
                                        <div class="col-md-6">
                                            <label for="inputCity" class="form-label">City</label>
                                                <input type="text" class="form-control" id="inputCity"  defaultValue={cityDisplay} onChange={e => setCity(e.target.value)}/>
                                        </div>

                                        <div class="col-md-4">
                                            <label for="inputState" class="form-label">State</label>
                                                <select id="inputState" class="form-select">
                                                    <option selected>Choose...</option>
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

                                        <div class="col-md-2">
                                            <label for="inputZip" class="form-label">Zip</label>
                                                <input type="text" class="form-control" id="inputZip"  defaultValue={zipDisplay} onChange={e => setZip(e.target.value)}/>
                                        </div>
                                    </div>


                                    

                                        {/* <div class="col-12">
                                            <div class="form-check">
                                                <input class="form-check-input" type="checkbox" id="gridCheck"/>
                                                    <label class="form-check-label" for="gridCheck">
                                                        Check me out
                                                    </label>
                                            </div>
                                        </div> */}{/* 
                                        <div class="col-12">
                                            <button type="submit" class="btn btn-primary">Sign in</button>
                                        </div> */}

                                </div>

                                <div>
                                    <h5 className='custom-landing-text pt-5'>Skills and Interests</h5>
                                        <p className='ps-3 mt-2'>Our application uses the information you enter here to match you to relevant volunteer opportunities around your area. Don't be shy to include any skills you may have as well as areas you are interested in volunteering.</p>
                                </div>

                                <div className='container pt-1 custom-form-section custom-form-blurb-color'>
                                    <div className='row g-3 pb-3'>
                                        <div class="col">
                                            <label for="otherinterests" class="form-label"></label>
                                                <textarea class="form-control" id="otherinterests" rows="3"  defaultValue={descriptionDisplay} onChange={e => setInterests(e.target.value)}></textarea>
                                        </div>
                                    </div>

                                </div>

                                <div class="col-12">
                                    <button type="submit" className="btn btn-primary custom-btn-post-color rounded-pill shadow" onClick={handleSubmit}>Save Changes</button>
                                </div>

                    
                            </form>
                        </div>
                    </div>
                </div>
        </div>






    );
}
}

export default StudentProfile;



        // <div>
        //     <div className='container-fluid mt-5 custom-landing-logo'>
        //         <a className="navbar-brand me-4 d-flex" href="#">
        //             <i className="bi bi-person-arms-up d-inline-block align-text-top fs-3 custom-icon-color pb-2"></i>
        //             NUVolunteers!
        //         </a>
        //     </div>

        //     <div className='container-fluid mt-5 text-center'>
        //         <h1 className='custom-landing-text'>Please fill out the following information:</h1>
        //     </div>

        //     <div className='container-fluid mt-5 text-center'>
        //         <form>
        //             <div className="mb-3">
        //                 <label for="exampleInputEmail1" className="form-label">Email address</label>
        //                 <input type="email" className="form-control" id="exampleInputEmail1" aria-describedby="emailHelp"/>
        //                 <div id="emailHelp" className="form-text">We'll never share your email with anyone else.</div>
        //             </div>
        //             <div className="mb-3">
        //                 <label for="exampleInputPassword1" className="form-label">Password</label>
        //                 <input type="password" className="form-control" id="exampleInputPassword1"/>
        //             </div>
        //             <div className="mb-3 form-check">
        //                 <input type="checkbox" className="form-check-input" id="exampleCheck1"/>
        //                 <label className="form-check-label" for="exampleCheck1">Check me out</label>
        //             </div>
        //             <button type="submit" className="btn btn-primary">Submit</button>
        //         </form>
        //      </div>
        // </div>