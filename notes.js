// import React, {useEffect, useState } from 'react';
// // Removed invalid Python import statement. If VolunteerRecommender is needed, implement or import it properly in JavaScript.


// function Posts() {
//   //get authentication and userid
//   const authToken = localStorage.getItem('authToken');
//   let userAuthData = null;
//   let userId = null;
//   const storedUserAuthData = localStorage.getItem('userAuthData');
//   const [interests, setInterests] = useState('');
  
//   if (storedUserAuthData) 
//       {
//       try {
//           userAuthData = JSON.parse(storedUserAuthData);
//           userId = userAuthData?.id;
//           } 
//       catch (error) {
//           console.error("Error parsing userAuthData:", error);
//           }
//       }
//   // console.log("Auth Token:", authToken);
//   console.log("User ID:", userId);

//   //function to apply a post
//   const handleApply = async (jobId) => {
//     setAppliedJobs(prevAppliedJobs => [...prevAppliedJobs, jobId]);
//     handleLike(jobId)
//   }
//   //function to like a post
//   const handleLike = async (jobId) => {
//     console.log("logging the initial likedjobs here", likedJobs);

//     console.log(`About to like job with ID: ${jobId}`);
//     if (likedJobs.includes(jobId)) {
//       console.log(`Job ID ${jobId} is already liked in this session.`);
//       return; // Don't proceed with the API call or state update
//       }
//     console.log(`Job id not yet liked so will proceed with api call`);

//     try{ //userid and jobid --> message saying done!
//       const res = await fetch(`${backendApiUrl}/interests/`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json'},
//         body: JSON.stringify({
//           user_id: userId,
//           job_id: jobId
//         })
//       });

//       if (res.ok)
//       {
//         const data = await res.json();
//         setLikedJobs(prevLikedJobs => [...prevLikedJobs, jobId]);//add to the list of liked
//         console.log("Job id is now LIKED!")

//       }
//       else
//       {
//         const errorData = await res.json();
//         throw new Error(errorData.error || 'Liking Action Failed');
            
//       }

//     }
//     catch(error){
//       console.error('Liking Action Faileed')

//     }
//   };

//   //likedjobs
//   const [likedJobs, setLikedJobs] = useState([]);
//   const [appliedJobs, setAppliedJobs] = useState([]);
//   const isJobLiked = (jobid) => {
//     return likedJobs.includes(jobid)
//   };//fn for is job liked?
//   const isJobApplied = (jobid) => {
//     return appliedJobs.includes(jobid)
//   }



  

//   const [expandedJobId, setExpandedJobId] = useState(null); // init all cards as non-expanded
//   const toggleExpand = (id) => {
//     setExpandedJobId(prevId => (prevId === id ? null : id));
//   }
//   const [allJobs, setAllJobs] = useState([]);
//   const [loading, setLoading] = useState(true);//loading state
//   const [error, setError] = useState(null);//any error
//   const [currentPage, setCurrentPage] = useState(1);//for pagination
//   const itemsPerPage = 6;//first index

 

//   //const backendApiUrl = 'https://raiso-ieee-2025.onrender.com';
//   const backendApiUrl ='https://backend-ieee.onrender.com';
//   //const backendApiUrl = 'http://localhost:5001'; LOCAL URLS FOR LOCAL DEPLOY
//   //const fastApiUrl = 'http://127.0.0.1:8000';
//   //const fastApiUrl = 'https://ieee-fastapi2.onrender.com';

//   const backupJobs = async () => {
//     console.log("GETTING BACKUP JOBS SINCE ONE OF THE MODELS FAILED");
//     setLoading(true);
//     setError(null);

//     try
//       {
//         const res = await fetch (`${backendApiUrl}/posts/?limit=250`)

//         if (!res.ok){
//              throw new Error(`HTTP error! status: ${res.status}`);
//         }
           
//         const data = await res.json();
//         const prioritizedJobs = data.sort((a, b) => {
//           const sponsor = "MEALS ON WHEELS NORTHEASTERN ILLINOIS";
//           const isA = a.organization === sponsor;
//           const isB = b.organization === sponsor;

//           if (isA && !isB) return -1;
//           if (!isA && isB) return 1;
//           return 0; // keep existing order if both are same type
//         }); 

//         setAllJobs(prioritizedJobs);
//       }
//       catch(error)
//       {
//         setError(error.message);//set error to be that message
//         setAllJobs([]);
//       }
//       finally{
//         setLoading(false);
//       }
// }
  

//   useEffect(()  => {

//       setLoading(true);
//       setError(null);

      

//       const fetchAllJobs = async () => {
//       console.log("SENDING GENERATE")
//       try
//       {
//         // const timeoutPromise = new Promise((_, reject) =>
//         //   setTimeout(() => reject(new Error("Request timed out")), 5000) // 5 seconds
//         // );
//         // const res = await fetch(`${fastApiUrl}/recommend/`,
//         // {
//         //   method: 'POST',
//         //   headers: { 'Content-Type' : 'application/json'},
//         //   body : JSON.stringify({userid : userId})
//         // }
//         // );
//         // if (!res.ok){
//         //       throw new Error(`HTTP error! status: ${res.status}`);
//         //     }
//         // console.log("SENT GENERATE")
      
//         // const data = await res.json();
//         // console.log("DATA")
//         // console.log(data)
//         // const jobs = data.jobs
//         // console.log(jobs)
//         // const prioritizedJobs = jobs.sort((a, b) => {
//         // const sponsor = "MEALS ON WHEELS NORTHEASTERN ILLINOIS";
//         // const isA = a.organization === sponsor;
//         // const isB = b.organization === sponsor;

//         // if (isA && !isB) return -1;
//         // if (!isA && isB) return 1;
//         // return 0;}); // keep existing order if both are same type

//         // setAllJobs(prioritizedJobs);
//         await backupJobs();

//         }
//         catch (error)
//         {
//           console.log("Couldn't generate recommended jobs.")
//           console.log(error.message)
//           //setError(error.message);//set error to be that message
//           setAllJobs([]);
//           //await backupJobs();//call backupJobs

//         }
//         finally{
//             setLoading(false);
//           }
//         };
//         fetchAllJobs();
//       },
//       [userId]);

//   const startIndex = (currentPage - 1) * itemsPerPage;//first index
//   const endIndex = startIndex + itemsPerPage;// lastindex
//   const currentJobs = allJobs.slice(startIndex, endIndex);
//   const totalPages = Math.ceil(allJobs.length / itemsPerPage);//number of pages

//   //moving to the next page
//   const handlePageChange = (newPage) => {
//     if (newPage >= 1 && newPage <= totalPages) {
//       setCurrentPage(newPage);
//     }
//   };

//   // const handleGenerate = async (e) => {//CASEY
    
//   // };

//   if (loading) {
//     return <div className='container mt-4 text-center'>Loading volunteer opportunities...</div>;
//   }

//   if (error) {
//     return <div className='container mt-4 text-center'>Error loading volunteer opportunities: {error}</div>;
//   }

//   return (

//     <div className='container mt-4'>
//       <div className='row'>
//           <div className='col-12 col-md-7'>
//               <div className='pt-4 text-center'>
//                 <h2> Volunteer Opportunities</h2>
//               </div>

//               <div className='d-flex justify-content-center gap-5 flex-wrap mt-3'>
//                 <button className={` btn btn-primary custom-btn-post-color mx-5 ${currentPage === 1 ? 'disabled': ''}`} onClick={() => handlePageChange(currentPage - 1)}> Previous</button>
//                 <button className={`btn btn-primary custom-btn-post-color mx-5 ${currentPage === totalPages ? 'disabled': ''}`} onClick={() => handlePageChange(currentPage + 1)}> Next</button>
//               </div>

//               <div className='pt-4 d-flex flex-column align-items-center row-gap-4'>
//                 {currentJobs.map((job)  => (     
                
//                       <div className={`card custom-card custom-card-bg-color ${job.organization === 'MEALS ON WHEELS NORTHEASTERN ILLINOIS' ? 'highlight-meals' : ''}`} key={job.id}>

//                       <div className="card-header card-header-bg-color" >
//                         {job.title}
//                       </div>
//                       <div className="card-body">
//                         <div className='d-flex justify-content-between'>
//                           <h5 className="card-title" style={{ cursor: 'pointer', textDecoration: 'underline' }} onClick={() => toggleExpand(job.id)} >{job.organization}</h5>
//                           <h6>{job.location}</h6>
//                         </div>
//                         <p className="card-text">{job.description} </p>
//                         {expandedJobId === job.id && (
//                           <>
//                           <p className="card-text mt-2"><b>Eligibility:</b> {job.requirement}</p>
//                           <p className="card-text"><b>Skills:</b> {job.skills}</p>
//                           </>
//                         )} 
//                         <div className='d-flex'>
//                           <div className='col-9'>
//                             <a href={job.url} className="btn btn-primary custom-btn-post-color" target="_blank"  rel="noopener noreferrer"  style= {{cursor: 'pointer', backgroundColor: isJobLiked(job.id) && isJobApplied(job.id) ? '#B1DABE' : '', color: isJobLiked(job.id) && isJobLiked(job.id) ? 'black' : '',}} onClick={() => handleApply(job.id)}>
//                               {isJobLiked(job.id) && isJobApplied(job.id) ? 'Applied!' : 'Apply'}
//                             </a>
//                           </div>
//                           <div className='col-3 d-flex justify-content-around'>
//                             <i className={`bi bi-hand-thumbs-up fs-3 ${isJobLiked(job.id) ? 'liked' : ''}`} style= {{cursor: 'pointer'}} onClick={() => handleLike(job.id)}></i>
//                             {/* <i className="bi bi-hand-thumbs-down"></i> */}
//                           </div>
//                         </div>
//                       </div>
//                     </div>
//                   ))} 
//               </div>
//             </div>
              
          

//           <div className='col-12 col-md-5 vh-100'>
//             <div className='row text-center h-40'> 
//               {/* <i className="bi bi-person-circle"  style='font-size: 50em;'></i> */}
//               <div className="icon-container" style={{width: "100%", height:"100%"}}>
//                 <i className="bi bi-person-circle" style={{fontSize: "10em", color: '#8CABF7'}}></i>
//               </div>
//             </div>

//             <div className='row'>
//                 <button className='btn btn-secondary btn-poster '>Customize your Posts!</button>
//             </div>

//             <div className='row'>
//                   {/*  */}

//                   <form className='pt-4'>
//                         <div>
//                             <h5 className='custom-landing-text mb-0'></h5>
//                             <p className='ps-3'>Use this space to search for any specific interests you may have and our model will recommend them for you!</p>
//                         </div>

//                         <div className='container pt-1 custom-form-section custom-form-blurb-color'>
//                             <div class="row g-3 pb-3">
//                                 <div class="col">
//                                     <label for="title" className="form-label">Title</label>
//                                         <input className="form-control" placeholder='What role would you like to do?' id="title" />
//                                 </div>
//                             </div>


//                             <div class='row g-3 pb-3'>
//                                 <div class="col">
//                                     <label for="org" class="form-label">Organization</label>
//                                         <input class="form-control" placeholder='Any organization in mind?' id="org"/>
//                                 </div>

//                             </div>
                            
//                             <div className='row g-3 pb-3'>
//                                 <div class="col-md-7">
//                                     <label for="city" class="form-label">City</label>
//                                         <input type="text"  placeholder='Location preferences?' class="form-control" id="city" />
//                                 </div>

//                                 <div class="col-md-5">
//                                     <label for="inputState" class="form-label">State</label>
//                                         <select id="inputState" class="form-select" >
//                                             <option selected>Choose...</option>
//                                             <option value="AL">Alabama</option>
//                                             <option value="AK">Alaska</option>
//                                             <option value="AZ">Arizona</option>
//                                             <option value="AR">Arkansas</option>
//                                             <option value="CA">California</option>
//                                             <option value="CO">Colorado</option>
//                                             <option value="CT">Connecticut</option>
//                                             <option value="DE">Delaware</option>
//                                             <option value="FL">Florida</option>
//                                             <option value="GA">Georgia</option>
//                                             <option value="HI">Hawaii</option>
//                                             <option value="ID">Idaho</option>
//                                             <option value="IL">Illinois</option>
//                                             <option value="IN">Indiana</option>
//                                             <option value="IA">Iowa</option>
//                                             <option value="KS">Kansas</option>
//                                             <option value="KY">Kentucky</option>
//                                             <option value="LA">Louisiana</option>
//                                             <option value="ME">Maine</option>
//                                             <option value="MD">Maryland</option>
//                                             <option value="MA">Massachusetts</option>
//                                             <option value="MI">Michigan</option>
//                                             <option value="MN">Minnesota</option>
//                                             <option value="MS">Mississippi</option>
//                                             <option value="MO">Missouri</option>
//                                             <option value="MT">Montana</option>
//                                             <option value="NE">Nebraska</option>
//                                             <option value="NV">Nevada</option>
//                                             <option value="NH">New Hampshire</option>
//                                             <option value="NJ">New Jersey</option>
//                                             <option value="NM">New Mexico</option>
//                                             <option value="NY">New York</option>
//                                             <option value="NC">North Carolina</option>
//                                             <option value="ND">North Dakota</option>
//                                             <option value="OH">Ohio</option>
//                                             <option value="OK">Oklahoma</option>
//                                             <option value="OR">Oregon</option>
//                                             <option value="PA">Pennsylvania</option>
//                                             <option value="RI">Rhode Island</option>
//                                             <option value="SC">South Carolina</option>
//                                             <option value="SD">South Dakota</option>
//                                             <option value="TN">Tennessee</option>
//                                             <option value="TX">Texas</option>
//                                             <option value="UT">Utah</option>
//                                             <option value="VT">Vermont</option>
//                                             <option value="VA">Virginia</option>
//                                             <option value="WA">Washington</option>
//                                             <option value="WV">West Virginia</option>
//                                             <option value="WI">Wisconsin</option>
//                                             <option value="WY">Wyoming</option>
//                                         </select>
//                                 </div>

//                             </div>
//                         </div>

//                         <div>
//                             <h5 className='custom-landing-text pt-5'>Skills and Interests</h5>
//                                 <p className='ps-3 mt-2'>Our application uses the information you enter here to match you to relevant volunteer opportunities around your area. Don't be shy to include any skills you may have as well as areas you are interested in volunteering.</p>
//                         </div>

//                         <div className='container pt-1 custom-form-section custom-form-blurb-color'>
//                             <div className='row g-3 pb-3'>
//                                 <div class="col">
//                                     <label for="interests" class="form-label"></label> 
//                                         <textarea class="form-control" id="interests" rows="3" value={interests} onChange={(e) => setInterests(e.target.value)}></textarea>
//                                 </div>
//                             </div>

//                             <div class="col-12">
//                                     <button type="submit" className="btn btn-primary custom-btn-post-color" >Generate</button>
//                             </div>

//                         </div>

              
//                     </form>

//                   {/*  */}
//             </div>

//           </div>
//       </div>
//     </div>


//   );
// };

// export default Posts;



















/////////////////////////////////



// import express from 'express';
// import supabase from '../supabaseClient.js';

// const router = express.Router();

// router.post('/', async (req, res) => {
//   const { user_id, job_id } = req.body;
//   //console.log("Received user_id:", user_id, "and job_id:", job_id, " (Type:", typeof job_id, ")");


//   //to check if the entry already exists
//   const { data: existingInterest, error: selectError } = await supabase
//       .from('user_interests')
//       .select('*')
//       .eq('user_id', user_id)
//       .eq('job_id', job_id)
//       .maybeSingle(); // Assuming you expect at most one match

//     // console.log("data:", existingInterest);
//     // console.log("error:", selectError);

//     if (selectError) {
//       console.error("Error checking existing interest:", selectError);
//       return res.status(500).json({ error: "Failed to check existing interest" });
//     }

//     if (existingInterest) {
//       // An entry already exists, so don't insert again
//       console.log("interest already recorded");
//       return res.status(200).json({ message: 'Interest already recorded' }); // Or a different status code/message
//     }

//   const { data, error } = await supabase
//     .from('user_interests')
//     .insert([{ user_id, job_id }]);

//   if (error) return res.status(500).json({ error: error.message });
//   res.status(201).json({ message: 'Interest added' });
// });

// router.delete('/', async (req, res) => {
//   const { user_id, job_id } = req.body;

//   const { error } = await supabase
//     .from('user_interests')
//     .delete()
//     .match({ user_id, job_id });

//   if (error) return res.status(500).json({ error: error.message });
//   res.json({ message: 'Interest removed' });
// });

// router.get('/:user_id', async (req, res) => {
//   const { user_id } = req.params;

//   const { data, error } = await supabase
//     .from('user_interests')
//     .select('jobs(*)')
//     .eq('user_id', user_id);

//   if (error) return res.status(500).json({ error: error.message });
//   res.json(data.map((entry) => entry.jobs));
// });

// export default router;
