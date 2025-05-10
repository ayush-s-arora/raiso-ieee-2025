import express from 'express';
import supabase from '../supabaseClient.js';

const router = express.Router();

// router.post('/signup', async (req, res) => {
//     const {
//         id,
//         email,
//         description,
//         first_name,
//         last_name,
//         city,
//         state,
//         zipcode
//       } = req.body;
    
//       if (!id || !email || !description) {
//         return res.status(400).json({ error: 'Missing required fields' });
//       }
    
//       const { data, error } = await supabase.from('users').insert([{
//         id,
//         email,
//         description,
//         first_name,
//         last_name,
//         city,
//         state,
//         zipcode
//       }]);
    
//       if (error) return res.status(500).json({ error: error.message });
    
//       res.status(201).json({ user: data[0] });
//     });

router.post('/signup', async (req, res) => {
  console.log('Received POST request at /auth/signup'); 
  const {
    email,
    password,
    firstName,
    lastName,
    city,
    state,
    zipcode,
    description
  } = req.body;

  console.log({
    email,
    password,
    firstName,
    lastName,
    city,
    state,
    zipcode,
    description
  });

  if (!email || !password) {
    console.log("Missing email or password");
    return res.status(400).json({error: "Email and password are needed"});
  }

  try {
    // Add user to the auth table
    const {data: authData, error: authError} = await supabase.auth.signUp({
      email: email,
      password: password
    });

    console.log("Logging stuff below");

    if (authError) {
      if (authError.message.includes('User already registered')) {
        return res.status(400).json({error: "You have already signed up. Try resetting your password."});
      }
      
      console.log("ERROR IN ADDING USER TO AUTH TABLE: ", authError.message);
      return res.status(400).json({error: authError.message});
    }

    if (!authError) {
      console.log("USER ADDED TO AUTH TABLE");
    }

    const userId = authData.user.id;
    console.log(userId);

    // Insert into users table (Parallelized)
    const userInsertPromise = supabase.from('users').insert([
      {
        userauth_id: userId, 
        email: email,
        first_name: firstName,
        last_name: lastName,
        city: city,
        state: state,
        zipcode: zipcode,
        description: description,
      }
    ]).select(); // Select new row

    // Wait for both operations to complete
    const [userData, userError] = await Promise.all([userInsertPromise]);

    if (userError) {
      console.log('User table insert error:', userError);
      return res.status(500).json({ error: "Try logging in on the login page" });
    }

    // Return success
    res.status(201).json({
      user: userData[0] // Sending the inserted user data back
    });

    if (!userError) {
      console.log("User added to users table");
    }

  } catch (error) {
    console.error('Error during signup:', error);
    res.status(500).json({ error: 'Internal server error during signup' });
  }
});


// /////////////////////////////
// router.post('/signup', async (req, res) => {
//   console.log('Received POST request at /auth/signup'); 
//   const {
//         email,
//         password,
//         firstName,
//         lastName,
//         city,
//         state,
//         zipcode,
//         description
//     } = req.body;//from the frontend

//   console.log({
//     email,
//         password,
//         firstName,
//         lastName,
//         city,
//         state,
//         zipcode,
//         description
//   })

//   if (!email || !password){
//     console.log("Missing email or password");
//     return res.status(400).json({error: "Email and password are needed"})
//   }

//   try
//   {
    
//     //add user to the auth table
//     const {data: authData, error: authError} = await supabase.auth.signUp({
//       email: email,
//       password: password
//     })
  
//     console.log("logging stuff below")

//     if (authError)//in case of authentication error
//     {
//       if (authError.message.includes('User already registered'))
//       {
//         return res.status(400).json({error: "You have already signed up. Try resetting your password."});
//       }
      
      
//         console.log("ERROR IN ADDING USER TO AUTH TABLE: ", authError.message)
//         return res.status(400).json({error: authError.message})
      
//     }

//     if (!authError){
//       console.log("USER ADDED TO AUTH TABLE")
//     }

//     const userId = authData.user.id
//     console.log(userId)
//     //insert into users table'
//     const {data: userData, error: userError} = await supabase
//             .from('users')
//             .insert([
//               {
//                 userauth_id: userId, // Use the ID from Supabase Auth
//                 email: email,
//                 first_name: firstName,
//                 last_name: lastName,
//                 city: city,
//                 state: state,
//                 zipcode: zipcode,
//                 description: description,
//               }
//             ])
//             .select();//select new row 
//     console.log("Insert result:", { userData, userError });

      

//     if (userError) {
//             // If there's an error inserting into the 'users' table,
//             console.log('User table insert error:', userError);
//             return res.status(500).json({ error: userError.message });
//         }///POTENTIAL BUG: ROLLING BACK?

//     res.status(201).json({
//       user: userData[0]
//     })

//     if (!userError){
//       console.log("User added to users table")
//     }
    

//   }

//   catch(error)
//   {
//     console.error('Error during signup:', error);
//     res.status(500).json({ error: 'Internal server error during signup' });
  
//   }

// });

///////////////////////////
//login
router.post('/login', async (req, res) => {
    console.log("LOGGING IN")
    const { email, password } = req.body;

    if (!email || !password) {
        return res.status(400).json({ error: 'Email and password are required' });
    }

    try 
    {
      const { data, error } = await supabase.auth.signInWithPassword({  // Changed to signInWithPassword
            email,
            password,
        });

      if (error) {
            console.log("Error logging in")
            return res.status(401).json({ error: error.message }); //  401 for invalid credentials
        }

      //  Successful login, get the user profile info.
      console.log("SUCCESSFUL LOGIN")
      res.json({ 
          user: data.user,  //  Include the user object
          token: data.session.access_token, //  Send the access token
      });

    }

    catch(error)
    {
      console.error('Error during login:', error);
      res.status(500).json({ error: 'Internal server error' });
    }


});


router.get('/me', async (req, res) => {
  //console.log("GETTING PROFILE")
  const { authorization } = req.headers;
  if (!authorization) return res.status(401).json({ error: 'Missing token' });

  const token = authorization.replace('Bearer ', '');
  //console.log("token is:", token)
  
  const {
    data,
    error
  } = await supabase.auth.getUser(token);

  
  // console.log("error", error)
  // console.log("data", data)

  if (error || !data) return res.status(401).json({ error: 'Invalid token' });

  let val = data.user.id.trim();
  //console.log("id", val)
  //console.log("Type of data.user.id before Supabase query:", typeof val); // Added console log
  const { data: profiledata, error: profileerror } = await supabase
    .from('users')
    .select('*')
    .eq('userauth_id', val);
    

  // console.log("PROFILE:", profiledata)
  if (profileerror) return res.status(500).json({ error: profileerror.message });
  // console.log("PROFILE:", profiledata)
  // console.log("Perror:", profileerror)
  res.json({ user: profiledata });
});

router.post('/updateme', async (req, res) => {
  console.log("Updating profile");

  const { authorization } = req.headers;
  if (!authorization) return res.status(401).json({ error: 'Missing token' });

  const token = authorization.replace('Bearer ', '');
  //console.log("token is:", token)
  
  const {
    data,
    error
  } = await supabase.auth.getUser(token);

  
  console.log("error", error)
  console.log("data", data)

  if (error || !data) return res.status(401).json({ error: 'Invalid token' });

  let val = data.user.id.trim();
  console.log("id", val)

  const {
        email,
        firstName,
        lastName,
        city,
        state,
        zipcode,
        description
    } = req.body;//from the frontend

  try
  {
    const {data: userData, error: userError} = await supabase
      .from('users') // Replace 'users' with the actual name of your public users table
      .update({
        email: email,
        first_name: firstName,
        last_name: lastName,
        city: city,
        state: state,
        zipcode: zipcode,
        description: description,
      })
      .eq('userauth_id', val) // Ensure you only update the record matching the logged-in user's ID
      .select(); // Optional: Select the updated data to send back


    if (userError)
    {
      console.error("Error updating user in public.users table:", userError);
      return res.status(500).json({ error: 'Failed to update profile' });
    }

    if (userData)
    {
      console.log("User profile updated in public.users:", userData[0]);
      return res.status(200).json({ message: 'Profile updated successfully', user: userData[0] });

    }
  }

  catch(error)
  {
    console.error('Error during profile update:', error);
    return res.status(500).json({ error: 'Internal server error' });
  
  }
})


router.delete('/delete-account', async (req, res) => {
  const { authorization } = req.headers;
  if (!authorization) return res.status(401).json({ error: 'Missing token' });

  const token = authorization.replace('Bearer ', '');

  // Get user from token
  const { data: userData, error: userError } = await supabase.auth.getUser(token);
  if (userError || !userData?.user?.id) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }

  const userId = userData.user.id;

  try {
    // 1. Delete from 'users' table (or any other user-related tables)
    const { error: dbDeleteError } = await supabase
      .from('users')
      .delete()
      .eq('userauth_id', userId);

    if (dbDeleteError) {
      console.error('Failed to delete user data from users table:', dbDeleteError);
      return res.status(500).json({ error: 'Failed to delete user data' });
    }

    // 2. Delete from auth.users using service role key
    const { createClient } = await import('@supabase/supabase-js');
    const adminSupabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY // Keep this safe, do NOT expose to client
    );

    const { error: authDeleteError } = await adminSupabase.auth.admin.deleteUser(userId);

    if (authDeleteError) {
      console.error('Failed to delete user from auth:', authDeleteError);
      return res.status(500).json({ error: 'Failed to delete user from auth system' });
    }

    return res.status(200).json({ message: 'Account deleted successfully' });

  } catch (err) {
    console.error('Error during account deletion:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});


export default router;
