const { Pool } = require('pg');
const dotenv = require('dotenv');

dotenv.config();

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
});

pool.connect()
    .then(() => console.log('PostgreSQL connected'))
    .catch(err => console.error('Database connection error', err));

module.exports = pool;
