import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './Admin.css'; 

const AdminScreen = () => {
  const [faqs, setFaqs] = useState([]);
  const [editingFaq, setEditingFaq] = useState(null);
  const [category, setCategory] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

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

  return (
    <div className="admin-container">
      <h2>Admin Dashboard</h2>
      <h3>Frequently Asked Questions</h3>
      
      <input 
        type="text" 
        placeholder="Search FAQs..." 
        value={searchTerm} 
        onChange={(e) => setSearchTerm(e.target.value)} 
      />

      <button onClick={() => setIsAdding(!isAdding)}>
        {isAdding ? 'Cancel' : 'Add Data'}
      </button>


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

      <button onClick={handleTrainData}>Train Data</button>

      {filteredFaqs.length === 0 ? (
        <p>No FAQs available.</p>
      ) : (
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
            {filteredFaqs.map(faq => (
              <tr key={faq.id}>
                <td>{faq.id}</td>
                <td>{faq.category}</td>
                <td>{faq.question}</td>
                <td>{faq.answer}</td>
                <td>
                  <button onClick={() => handleEdit(faq)}>Edit</button>
                  <button onClick={() => handleDelete(faq.id)}>Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
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
    </div>
  );
};

export default AdminScreen;
