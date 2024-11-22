import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './styles/Admin.css'; 
import '@fortawesome/fontawesome-free/css/all.min.css';

const AdminScreen = () => {
  const [faqs, setFaqs] = useState([]);
  const [editingFaq, setEditingFaq] = useState(null);
  const [category, setCategory] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [pendingAdmins, setPendingAdmins] = useState([]);
  const [showNotifications, setShowNotifications] = useState(false);

  
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 8; 

  useEffect(() => {
    fetchFaqs();
  }, []);

  const fetchFaqs = async () => {
    try {
      const response = await axios.get('http://192.168.11.188:5000/faqs');
      setFaqs(response.data);
    } catch (error) {
      console.error("Error fetching FAQs:", error);
    }
  };


  const handleEdit = (faq) => {
    setEditingFaq(faq.id);
    setCategory(faq.category);
    setQuestion(faq.question);
    setAnswer(faq.answer);
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`http://192.168.11.188:5000/faqs/${id}`);
      fetchFaqs();
    } catch (error) {
      console.error("Error deleting FAQ:", error);
    }
  };

  const handleSave = async () => {
    try {
      await axios.put(`http://192.168.11.188:5000/faqs/${editingFaq}`, {
        category,
        question,
        answer,
      });
      setEditingFaq(null);
      fetchFaqs();
      clearInputs();
    } catch (error) {
      console.error("Error editing FAQ:", error);
    }
  };

  const handleAdd = async () => {
    try {
      await axios.post('http://192.168.11.188:5000/faqs', {
        category,
        question,
        answer,
      });
      setIsAdding(false);
      fetchFaqs();
      clearInputs();
    } catch (error) {
      console.error("Error adding FAQ:", error);
    }
  };


  const handleTrainData = async () => {
    try {
      const response = await axios.post('http://192.168.11.188:5000/train');
      alert(response.data.message);
    } catch (error) {
      console.error("Error training data:", error);
      alert("Failed to start training. See console for details.");
    }
  };

  const clearInputs = () => {
    setCategory('');
    setQuestion('');
    setAnswer('');
  };

  const filteredFaqs = faqs.filter(faq => 
    faq.question.toLowerCase().includes(searchTerm.toLowerCase()) || 
    faq.category.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalRows = filteredFaqs.length;
  const totalPages = Math.ceil(totalRows / rowsPerPage);
  const currentRows = filteredFaqs.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );

  const nextPage = () => {
    if (currentPage < totalPages) setCurrentPage(currentPage + 1);
  };

  const previousPage = () => {
    if (currentPage > 1) setCurrentPage(currentPage - 1);
  };



  const fetchPendingAdmins = async () => {
    try {
      const response = await axios.get('http://192.168.11.188:5000/pending_admins');
      setPendingAdmins(response.data);
    } catch (error) {
      console.error("Error fetching pending admin requests:", error);
    }
  };

  const handleAdminAction = async (email, action) => {
    try {
      const response = await axios.post(`http://192.168.11.188:5000/${action}_admin`, { email });
      alert(response.data.message);
      fetchPendingAdmins(); 
    } catch (error) {
      console.error(`Error ${action} admin request:`, error);
      alert(`Failed to ${action} admin request.`);
    }
  };

  const toggleNotifications = () => {
    setShowNotifications(!showNotifications);
    if (!showNotifications) fetchPendingAdmins();
  };

  const handleLogout = () => {
    localStorage.removeItem('adminToken'); 
    window.location.href = '/adminlogin'; 
  };

  return (
    <div className="admin-container">
      <h2>Admin Dashboard</h2>
      <h3>Frequently Asked Questions</h3>

      <div className="button-notif">
        <button className="notif" onClick={toggleNotifications}>
          Notifications ({pendingAdmins.length})
        </button>
      </div>

      <div className="button-container">
      <input className="search"
        type="text" 
        placeholder="Search FAQs..." 
        value={searchTerm} 
        onChange={(e) => setSearchTerm(e.target.value)} 
      />
        <button className="but" onClick={handleTrainData}>Train Data</button>
        <button className="but" onClick={() => setIsAdding(!isAdding)}>
          {isAdding ? 'Cancel' : 'Add Data'}
        </button>
      </div>

      {showNotifications && (
        <div className="notification-panel">
          <h3>Pending Admin Approval Requests</h3>
          {pendingAdmins.length === 0 ? (
            <p>No pending admin requests.</p>
          ) : (
            <ul>
              {pendingAdmins.map(admin => (
                <li key={admin.email}>
                  <p>{admin.firstName} {admin.lastName} - {admin.department}</p>
                  <button onClick={() => handleAdminAction(admin.email, 'approve')}>Approve</button>
                  <button onClick={() => handleAdminAction(admin.email, 'decline')}>Decline</button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}


      {isAdding && (
        <div>
          <h3>Add New FAQ</h3>
          <input 
            type="text" 
            value={category} 
            onChange={(e) => setCategory(e.target.value)} 
            placeholder="Category" 
          />
          <input 
            type="text" 
            value={question} 
            onChange={(e) => setQuestion(e.target.value)} 
            placeholder="Question" 
          />
          <input 
            type="text" 
            value={answer} 
            onChange={(e) => setAnswer(e.target.value)} 
            placeholder="Answer" 
          />
          <button onClick={handleAdd}>Add FAQ</button>
        </div>
      )}

      

      {editingFaq && (
        <div>
          <h3>Edit FAQ</h3>
          <input 
            type="text" 
            value={category} 
            onChange={(e) => setCategory(e.target.value)} 
            placeholder="Category" 
          />
          <input 
            type="text" 
            value={question} 
            onChange={(e) => setQuestion(e.target.value)} 
            placeholder="Question" 
          />
          <input 
            type="text" 
            value={answer} 
            onChange={(e) => setAnswer(e.target.value)} 
            placeholder="Answer" 
          />
          <button onClick={handleSave}>Save</button>
          <button onClick={() => setEditingFaq(null)}>Cancel</button>
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Category</th>
            <th>Question</th>
            <th>Answer</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {currentRows.map(faq => (
            <tr key={faq.id}>
              <td>{faq.id}</td>
              <td>{faq.category}</td>
              <td>{faq.question}</td>
              <td>{faq.answer}</td>
              <td>
                <button className="edit" onClick={() => handleEdit(faq)}>Edit</button>
                <button className="delete" onClick={() => handleDelete(faq.id)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {totalRows > rowsPerPage && (
        <div className="pagination">
          <button onClick={previousPage} disabled={currentPage === 1} className="page-arrow">
            <i className="fas fa-arrow-left"></i> 
          </button>
          <span>Page {currentPage} of {totalPages}</span>
          <button onClick={nextPage} disabled={currentPage === totalPages} className="page-arrow">
            <i className="fas fa-arrow-right"></i> 
          </button>
        </div>
      )}
      <button onClick={handleLogout}>Logout</button>
    </div>
  );
};

export default AdminScreen;