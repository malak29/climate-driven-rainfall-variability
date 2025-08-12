import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const rainfallAPI = {
  // Get predictions
  getPredictions: async (district, month) => {
    const response = await api.post('/predict', { district, month });
    return response.data;
  },

  // Get risk assessment
  getRiskAssessment: async (district) => {
    const response = await api.get(`/risk-assessment?district=${district}`);
    return response.data;
  },

  // Get recommendations
  getRecommendations: async (riskLevel) => {
    const response = await api.get(`/recommendations?risk_level=${riskLevel}`);
    return response.data;
  },

  // Get historical data
  getHistoricalData: async (params) => {
    const response = await api.get('/historical-data', { params });
    return response.data;
  },
};

export default api;