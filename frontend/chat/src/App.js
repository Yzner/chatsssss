

// import React from 'react';
// import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
// import ChatbotScreen from './ChatbotScreen';
// import AdminScreen from './AdminScreen';

// const App = () => {
//   return (
//     <Router>
//       <Routes>
//         <Route path="/" element={<Navigate to="/chatbotscreen" />} />
//         <Route path="/chatbotscreen" element={<ChatbotScreen />} />
//         <Route path="/admin" element={<AdminScreen />} />
//         <Route path="*" element={<h2>404 Page Not Found</h2>} />
//       </Routes>
//     </Router>
//   );
// };

// export default App;





// import React from 'react';
// import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
// import ChatbotScreen from './ChatbotScreen';
// import AdminScreen from './AdminScreen';
// import UserLoginScreen from './UserLoginScreen';

// const App = () => {
//   const isAuthenticated = localStorage.getItem('token') !== null;

//   return (
//     <Router>
//       <Routes>
//         <Route path="/" element={<Navigate to="/login" />} />
//         <Route path="/login" element={<UserLoginScreen />} />
//         <Route
//           path="/chatbotscreen"
//           element={isAuthenticated ? <ChatbotScreen /> : <Navigate to="/login" />}
//         />
//         <Route path="/admin" element={<AdminScreen />} />
//         <Route path="*" element={<h2>404 Page Not Found</h2>} />
//       </Routes>
//     </Router>
//   );
// };

// export default App;


import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ChatbotScreen from './ChatbotScreen';
import UserLoginScreen from './UserLoginScreen';
import AdminLoginScreen from './AdminLoginScreen';
import AdminScreen from './AdminScreen'; 
import Main from "./website/Main";

const App = () => {
  const isAdminAuthenticated = localStorage.getItem('adminToken') !== null;
  const isAuthenticated = localStorage.getItem('token') !== null;

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Main />} />
        <Route path="/login" element={<UserLoginScreen />} />
        <Route
          path="/chatbotscreen"
          element={isAuthenticated ? <ChatbotScreen /> : <Navigate to="/login" />}
        />
        <Route path="/adminlogin" element={<AdminLoginScreen />} />
        <Route
          path="/adminscreen"
          element={isAdminAuthenticated ? <AdminScreen /> : <Navigate to="/adminlogin" />}
        />
        <Route path="*" element={<h2>404 Page Not Found</h2>} />
      </Routes>
    </Router>
  );
};

export default App;
