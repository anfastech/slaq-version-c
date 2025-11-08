-- SLAQ Database Setup Script
-- Run this script in PostgreSQL to create the database and user

-- Create the database
CREATE DATABASE slaq_db;

-- Connect to the database
\c slaq_db

-- Create user if using a different user (optional)
-- CREATE USER slaq_user WITH PASSWORD '123456789';
-- GRANT ALL PRIVILEGES ON DATABASE slaq_db TO slaq_user;

-- If using the default postgres user, just make sure the password is correct
-- You can change the postgres password with:
-- ALTER USER postgres WITH PASSWORD '123456789';
