import express from 'express';
import cors from 'cors';
import usersRouter from './routes/userRoutes.js';
import postsRouter from './routes/posts.js';
import interestsRouter from './routes/interests.js';
import supabase from './supabaseClient.js';
import authRoutes from './routes/auth.js';
import dotenv from 'dotenv';
dotenv.config();


const app = express();

app.use(cors());
app.use(express.json());

app.get('/auth/test', (req, res) => {
  res.send('Auth routes are being mounted!');
});
app.use('/auth', authRoutes);//authentication
app.use('/users', usersRouter);
app.use('/posts', postsRouter);
app.use('/interests', interestsRouter);

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));


app.get('/api/test-supabase', async (req, res) => {
    const { data, error } = await supabase.from('users').select('*').limit(1);
  
    if (error) return res.status(500).json({ error: error.message });
    res.json({ connected: true, sampleUser: data[0] || null });
  });