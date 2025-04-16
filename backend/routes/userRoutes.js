import express from 'express';
import supabase from '../supabaseClient.js';

const router = express.Router();

router.get('/', async (req, res) => {
  const { data, error } = await supabase.from('users').select('*');
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

router.get('/:id', async (req, res) => {
  const { id } = req.params;
  const { data, error } = await supabase.from('users').select('*').eq('id', id).single();
  if (error) return res.status(404).json({ error: 'User not found' });
  res.json(data);
});


router.put('/:id', async (req, res) => {
  const { id } = req.params;
  const { email, description } = req.body;
  const { data, error } = await supabase.from('users').update({ email, description }).eq('id', id);
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

router.delete('/:id', async (req, res) => {
  const { id } = req.params;
  const { error } = await supabase.from('users').delete().eq('id', id);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ message: 'User deleted successfully' });
});

export default router;
