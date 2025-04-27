import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ChatbotScreen from './ChatbotScreen';
import UserLoginScreen from './UserLoginScreen';
import SignUp from './SignUp';
import AdminLoginScreen from './AdminLoginScreen';
import AdminScreen from './AdminScreen'; 
import Main from "./website/Main";
import About from"./website/About";
import MissionVision from './website/AboutNav/MandV';
import GoalsObjective from './website/AboutNav/GandO';
import AcademicPrograms from './website/AboutNav/Programs';
import CollegeOrgan from './website/AboutNav/CollegeOrgan';
import StudentOrg from './website/AboutNav/StudentOrg';
import Services from './website/ServicesNav/Services';
import AcadAwards from './website/ServicesNav/AcadAwards';
import Procedures from './website/ServicesNav/Procedures';
import Enrollment from './website/ServicesNav/Enrollment';
import News from './website/News';
import EmailReq from './website/ServicesNav/EmailReq';
import ContactUs from './website/ContactUs';
import Adding from './website/Procedures/Adding';
import Changing from './website/Procedures/Changing';
import Completion from './website/Procedures/Comple';
import CrossEnrollment from './website/Procedures/CrossEnroll';
import Dropping from './website/Procedures/Dropping';
import GeneralClearance from './website/Procedures/GenClean';
import GoodMoral from './website/Procedures/GoodMo';
import Shifting from './website/Procedures/Shifting';
import AcceptanceShift from './website/Procedures/Shift';
import Substitution from './website/Procedures/Substitute';
import Mainn from './website/mainn';

const App = () => {
  const isAdminAuthenticated = localStorage.getItem('adminToken') !== null;
  const isAuthenticated = localStorage.getItem('token') !== null;

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Main />} />
        <Route path="/login" element={<UserLoginScreen />} />
        <Route path="/signup" element={<SignUp />} />
        <Route
          path="/chatbotscreen"
          element={isAuthenticated ? <ChatbotScreen /> : <Navigate to="/login" />}
        />
        <Route path="/adminlogin" element={<AdminLoginScreen />} />
        <Route
          path="/adminscreen"
          element={isAdminAuthenticated ? <AdminScreen /> : <Navigate to="/adminlogin" />}
        />
        <Route path="/about" element={<About />} />
        <Route path="/ContactUs" element={<ContactUs />} />
        <Route path="/MandV" element={<MissionVision />} />
        <Route path="/GandO" element={<GoalsObjective />} />
        <Route path="/Programs" element={<AcademicPrograms/>} />
        <Route path="/CollegeOrgan" element={<CollegeOrgan/>} />
        <Route path="/StudentOrg" element={<StudentOrg/>} />
        <Route path="/Services" element={<Services/>} />
        <Route path="/AcadAwards" element={<AcadAwards/>} />
        <Route path="/Procedures" element={<Procedures/>} />
        <Route path="/Enrollment" element={<Enrollment/>} />
        <Route path="/EmailReq" element={<EmailReq/>} />
        <Route path="/News" element={<News/>} />
        <Route path="/Adding" element={<Adding/>} />
        <Route path="/Comple" element={<Completion/>} />
        <Route path="/CrossEnroll" element={<CrossEnrollment/>} />
        <Route path="/Dropping" element={<Dropping/>} />
        <Route path="/GenClean" element={<GeneralClearance/>} />
        <Route path="/GoodMo" element={<GoodMoral/>} />
        <Route path="/Shifting" element={<Shifting/>} />
        <Route path="/Shift" element={<AcceptanceShift/>} />
        <Route path="/Substitute" element={<Substitution/>} />
        <Route path="/Changing" element={<Changing/>} />
        <Route path="/home" element={<Mainn/>} />
        <Route path="*" element={<h2>404 Page Not Found</h2>} />
      </Routes>
    </Router>
  );
};

export default App;
