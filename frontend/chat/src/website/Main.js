import React from "react";
import "../styles/Main.css"; 

const Main = () => {
  return (
    <div>
      {/* Navigation Bar */}
      <header className="navbar">
        <div className="logo">Palawan State University</div>
        <nav>
          <ul>
            <li><a href="#about">About</a></li>
            <li><a href="#organization">Organization</a></li>
            <li><a href="#programs">Degree Programs</a></li>
            <li><a href="#research">Research</a></li>
            <li><a href="#resources">Student Resources</a></li>
          </ul>
        </nav>
      </header>

      {/* Welcome Section */}
      <section className="welcome-section">
        <div className="welcome-text">
          <h1>Welcome to the College of Science</h1>
          <p>
            bjhbcsbkdfbfsdfsfsghghdhgf.
          </p>
          <button className="know-more-btn">Know More</button>
        </div>
      </section>

      {/* News and Events */}
      <section className="news-events">
        <h2>News and Events</h2>
        <div className="news-items">
          <div className="news-item">
            <img src="news1.jpg" alt="News 1" />
            <p>News Title 1</p>
          </div>
          <div className="news-item">
            <img src="news2.jpg" alt="News 2" />
            <p>News Title 2</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>© 2025 Palawan State University - College of Science</p>
      </footer>
    </div>
  );
};

export default Main;
