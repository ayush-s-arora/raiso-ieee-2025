import React, { useEffect, useState } from 'react';

function Posts() {
  const authToken = localStorage.getItem('authToken');
  const storedUserAuthData = localStorage.getItem('userAuthData');
  const [interests, setInterests] = useState('');
  const [userId, setUserId] = useState(null);

  const [title, setTitle] = useState('');
  const [org, setOrg] = useState('');
  const [city, setCity] = useState('');
  const [state, setState] = useState('');
  const [blurb, setBlurb] = useState('');

  //const backendApiUrl = 'https://backend-ieee.onrender.com';
  const backendApiUrl = 'http://localhost:5001'; 
  const [likedJobs, setLikedJobs] = useState([]);
  const [appliedJobs, setAppliedJobs] = useState([]);
  const [allJobs, setAllJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [expandedJobId, setExpandedJobId] = useState(null);

  const itemsPerPage = 6;
  const fastApiUrl = `http://127.0.0.1:8000`

  const [generatedJobs, setGeneratedJobs] = useState([]);
  const generateJobs = async (e) => {
    e.preventDefault();

    console.log("GENERATE JOBS");
    console.log("Blurb", blurb);
    try{
      const res = await fetch (`${fastApiUrl}/generate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            blurb : blurb
        }
      )
    })
    if (res.ok){
      const data = await res.json();
      setGeneratedJobs(data.jobs);
    }

    }
    catch(error){
      console.log("generate model not reacy")
    }
  }

  useEffect(() => {
    if (storedUserAuthData) {
      try {
        const userAuthData = JSON.parse(storedUserAuthData);
        setUserId(userAuthData?.id);
      } catch (error) {
        console.error("Error parsing userAuthData:", error);
      }
    }
  }, [storedUserAuthData]);

  useEffect(() => {
    const fetchLikedJobs = async () => {
      if (userId) {
        try {
          const res = await fetch(`${backendApiUrl}/interests/liked/ids/${userId}`);
          if (res.ok) {
            const data = await res.json();
            setLikedJobs(data);
          }
        } catch (error) {
          console.error("Error fetching liked jobs:", error);
        }
      }
    };

    const fetchAppliedJobs = async () => {
      if (userId) {
        try {
          const res = await fetch(`${backendApiUrl}/interests/applied/ids/${userId}`);
          if (res.ok) {
            const data = await res.json();
            setAppliedJobs(data);
          }
        } catch (error) {
          console.error("Error fetching applied jobs:", error);
        }
      }
    };

    fetchLikedJobs();
    fetchAppliedJobs();
  }, [userId]);

  const updateInterestInDB = async (jobId, actionType) => {
    if (userId) {
      try {
        const res = await fetch(`${backendApiUrl}/interests/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            job_id: jobId,
            type: actionType
          })
        });

        if (res.ok) {
          if (actionType === 'like' && !likedJobs.includes(jobId)) {
            setLikedJobs(prev => [...prev, jobId]);
          } else if (actionType === 'apply' && !appliedJobs.includes(jobId)) {
            setAppliedJobs(prev => [...prev, jobId]);
          }
        }
      } catch (error) {
        console.error(`${actionType} Action Failed`, error);
      }
    }
  };

  const handleApply = async (jobId) => {
    if (!appliedJobs.includes(jobId)) {
      await updateInterestInDB(jobId, 'apply');
    }
  };

  const handleLike = async (jobId) => {
    if (!likedJobs.includes(jobId)) {
      await updateInterestInDB(jobId, 'like');
    }
  };

  const isJobLiked = (jobId) => likedJobs.includes(jobId);
  const isJobApplied = (jobId) => appliedJobs.includes(jobId);

  const toggleExpand = (id) => {
    setExpandedJobId(prevId => (prevId === id ? null : id));
  };

  const backupJobs = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${backendApiUrl}/posts/?limit=250`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

      const data = await res.json();
      const prioritizedJobs = data.sort((a, b) => {
        const sponsor = "MEALS ON WHEELS NORTHEASTERN ILLINOIS";
        return (a.organization === sponsor ? -1 : b.organization === sponsor ? 1 : 0);
      });

      setAllJobs(prioritizedJobs);
    } catch (error) {
      setError(error.message);
      setAllJobs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchAllJobs = async () => {
      try {
        await backupJobs();
      } catch (error) {
        setAllJobs([]);
      } finally {
        setLoading(false);
      }
    };
    fetchAllJobs();
  }, [userId]);

  const filteredJobs = allJobs.filter(job => !appliedJobs.includes(job.id) || job.organization === 'MEALS ON WHEELS NORTHEASTERN ILLINOIS');
  const startIndex = (currentPage - 1) * itemsPerPage;
  const currentJobs = filteredJobs.slice(startIndex, startIndex + itemsPerPage);
  const totalPages = Math.ceil(filteredJobs.length / itemsPerPage);

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  if (loading) return <div className='container mt-4 text-center'>Loading volunteer opportunities...</div>;
  if (error) return <div className='container mt-4 text-center'>Error loading volunteer opportunities: {error}</div>;

  return (
    <div className='container mt-4'>
      <div className='row'>
        <div className='col-12 col-md-7'>
          <div className='pt-4 text-center'>
            <h2>Volunteer Opportunities</h2>
          </div>

          <div className='d-flex justify-content-center gap-5 flex-wrap mt-3'>
            <button className={`btn btn-primary custom-btn-post-color mx-5 ${currentPage === 1 ? 'disabled' : ''}`} onClick={() => handlePageChange(currentPage - 1)}>Previous</button>
            <button className={`btn btn-primary custom-btn-post-color mx-5 ${currentPage === totalPages ? 'disabled' : ''}`} onClick={() => handlePageChange(currentPage + 1)}>Next</button>
          </div>

          <div className='pt-4 d-flex flex-column align-items-center row-gap-4'>
            {(generatedJobs && generatedJobs.length > 0 ? generatedJobs : currentJobs).map((job) => (
              <div className={`card custom-card custom-card-bg-color ${job.organization === 'MEALS ON WHEELS NORTHEASTERN ILLINOIS' ? 'highlight-meals' : ''}`} key={job.id}>
                <div className="card-header card-header-bg-color">{job.title}</div>
                <div className="card-body">
                  <div className='d-flex justify-content-between'>
                    <h5 className="card-title" style={{ cursor: 'pointer', textDecoration: 'underline' }} onClick={() => toggleExpand(job.id)}>{job.organization}</h5>
                    <h6>{job.location}</h6>
                  </div>
                  <p className="card-text">{job.description}</p>
                  {expandedJobId === job.id && (
                    <>
                      <p className="card-text mt-2"><b>Eligibility:</b> {job.requirement}</p>
                      <p className="card-text"><b>Skills:</b> {job.skills}</p>
                    </>
                  )}
                  <div className='d-flex'>
                    <div className='col-9'>
                      <button
                        className="btn btn-primary custom-btn-post-color"
                        style={{
                          backgroundColor: isJobApplied(job.id) ? '#B1DABE' : '',
                          color: isJobApplied(job.id) ? 'black' : '',
                        }}
                        onClick={() => handleApply(job.id)}
                        disabled={isJobApplied(job.id)}
                      >
                        {isJobApplied(job.id) ? 'Applied!' : 'Apply'}
                      </button>
                    </div>
                    <div className='col-3 d-flex justify-content-around'>
                      <i className={`bi bi-hand-thumbs-up fs-3 ${isJobLiked(job.id) ? 'liked' : ''}`} style={{ cursor: 'pointer' }} onClick={() => handleLike(job.id)}></i>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className='col-12 col-md-5 vh-100'>
          <div className='row text-center h-40'>
            <div className="icon-container" style={{ width: "100%", height: "100%" }}>
              <i className="bi bi-person-circle" style={{ fontSize: "10em", color: '#8CABF7' }}></i>
            </div>
          </div>

          {/* <div className='row'>
            <button className='btn btn-secondary btn-poster'>Customize your Posts!</button>
          </div> */}

       
            {/* <form className='pt-4' onSubmit={generateJobs}>
              <div>
                <p className='ps-3'>Use this space to search for any specific interests you may have and our model will recommend them for you!</p>
              </div>

              <div className='container pt-1 custom-form-section custom-form-blurb-color'>
                <div className="row g-3 pb-3">
                  <div className="col">
                    <label htmlFor="title" className="form-label" id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}>Title</label>
                    <input className="form-control" placeholder='What role would you like to do?' id="title" />
                  </div>
                </div>

                <div className='row g-3 pb-3'>
                  <div className="col">
                    <label htmlFor="org" className="form-label" id="org"
              value={org}
              onChange={(e) => setOrg(e.target.value)}>Organization</label>
                    <input className="form-control" placeholder='Any organization in mind?' id="org" />
                  </div>
                </div>

                <div className='row g-3 pb-3'>
                  <div className="col-md-7">
                    <label htmlFor="city" className="form-label" id="city"
              value={city}
              onChange={(e) => setCity(e.target.value)}>City</label>
                    <input type="text" placeholder='Location preferences?' className="form-control" id="city" />
                  </div>

                  <div className="col-md-5">
                    <label htmlFor="inputState" className="form-label" id="inputState"
          
              value={state}
              onChange={(e) => setState(e.target.value)}>State</label>
                    <select id="inputState" className="form-select">
                      <option>Choose...</option>
                      {['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'].map(state =>
                        <option key={state} value={state}>{state}</option>
                      )}
                    </select>
                  </div>
                </div>
              </div>

              <div>
                <h5 className='custom-landing-text pt-5'>Skills and Interests</h5>
                <p className='ps-3 mt-2'>Our application uses the information you enter here to match you to relevant volunteer opportunities around your area.</p>
              </div>

              <div className='container pt-1 custom-form-section custom-form-blurb-color'>
                <div className='row g-3 pb-3'>
                  <div className="col">
                    <label htmlFor="blurb" className="form-label" id="blurb"
              rows="3"
              value={blurb}
              onChange={(e) => setBlurb(e.target.value)}>Your Interests</label>
                    <textarea className="form-control" id="interests" rows="3" value={interests} onChange={(e) => setInterests(e.target.value)}></textarea>
                  </div>
                </div>

                <div className="col-12">
                  <button type="submit" className="btn btn-primary custom-btn-post-color">Generate</button>
                </div>
              </div>
            </form> */}

<div className='row'>
  <button className='btn btn-secondary btn-poster'>Customize your Posts!</button>
</div>

<div className='row'>
  <form className='pt-4' onSubmit={generateJobs}>
    <div>
      <p className='ps-3'>Use this space to search for any specific interests you may have and our model will recommend them for you!</p>
    </div>

    <div className='container pt-1 custom-form-section custom-form-blurb-color'>
      <div className="row g-3 pb-3">
        <div className="col">
          <label htmlFor="title" className="form-label">Title</label>
          <input
            type="text"
            className="form-control"
            placeholder='What role would you like to do?'
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </div>
      </div>

      <div className='row g-3 pb-3'>
        <div className="col">
          <label htmlFor="org" className="form-label">Organization</label>
          <input
            type="text"
            className="form-control"
            placeholder='Any organization in mind?'
            id="org"
            value={org}
            onChange={(e) => setOrg(e.target.value)}
          />
        </div>
      </div>

      <div className='row g-3 pb-3'>
        <div className="col-md-7">
          <label htmlFor="city" className="form-label">City</label>
          <input
            type="text"
            placeholder='Location preferences?'
            className="form-control"
            id="city"
            value={city}
            onChange={(e) => setCity(e.target.value)}
          />
        </div>

        <div className="col-md-5">
          <label htmlFor="inputState" className="form-label">State</label>
          <select
            id="inputState"
            className="form-select"
            value={state}
            onChange={(e) => setState(e.target.value)}
          >
            <option>Choose...</option>
            {['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'].map(state =>
              <option key={state} value={state}>{state}</option>
            )}
          </select>
        </div>
      </div>
    </div>

    <div>
      <h5 className='custom-landing-text pt-5'>Skills and Interests</h5>
      <p className='ps-3 mt-2'>Our application uses the information you enter here to match you to relevant volunteer opportunities around your area.</p>
    </div>

    <div className='container pt-1 custom-form-section custom-form-blurb-color'>
      <div className='row g-3 pb-3'>
        <div className="col">
          <label htmlFor="blurb" className="form-label">Your Interests</label>
          <textarea
            className="form-control"
            id="blurb" // Changed id to match htmlFor
            rows="3"
            value={blurb} // Use the 'blurb' state
            onChange={(e) => setBlurb(e.target.value)} // Update the 'blurb' state
          ></textarea>
        </div>
      </div>

      <div className="col-12">
        <button type="submit" className="btn btn-primary custom-btn-post-color">Generate</button>
      </div>
    </div>
  </form>
</div>
          
        </div>
      </div>
    </div>
  );
}

export default Posts;
