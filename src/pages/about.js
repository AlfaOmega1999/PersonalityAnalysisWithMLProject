import React from 'react';
import 'chart.js/auto';

import "../style/about.css"

const About = () => {


  return (
    <header id="home">
      <div className="conatiner">
		<div className="wrap">
			<div className="aboutbox">
					
					<div className="about-poster pa">
					<h1 className='about-title'>About the app</h1>
					</div>
					<p className='about-text'>It is a free, open source application that uses machine learning techniques to determine an individual's personality based on the text messages they write.</p>
					<br />
					<p className='about-text'>The results obtained are based on AI and may not be fully accurate. The use of this application should not condition a person's decisions based on its results. </p>
					<br />
					<p className='about-text'>This application is an end-of-degree project for the degree in computer engineering taught at the Faculty of Computer Science of Barcelona (FIB), specialising in software engineering.</p>
				</div>
            </div>
        </div>
    </header>
  );
};

export default About;