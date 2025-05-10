import React from 'react';
import Navbar from './Navbar';
// import SearchBar from './SearchBar';
import Footer from './Footer'
import Posts from './Posts';


const Home = () => {
    return (
        <div>
            <Navbar />
            {/* <SearchBar /> */}
            <Posts />
            <Footer />
        </div>
    );
};

export default Home;