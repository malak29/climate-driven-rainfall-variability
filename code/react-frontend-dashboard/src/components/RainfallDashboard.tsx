import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';
import { AlertTriangle, TrendingDown, TrendingUp, Cloud, Droplets, AlertCircle, Map, Activity, Target, DollarSign, Users, Calendar, ChevronRight, Download, RefreshCw } from 'lucide-react';

const RainfallDashboard = () => {
  const [selectedRegion, setSelectedRegion] = useState('All India');
  const [selectedTimeframe, setSelectedTimeframe] = useState('Current Year');
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [alertVisible, setAlertVisible] = useState(true);
  const [realData, setRealData] = useState(null);

  // Simulated real-time data update
  const [lastUpdate, setLastUpdate] = useState(new Date().toLocaleString());

  const fetchData = async () => {
    try {
      setIsLoading(true);
      const predictions = await rainfallAPI.getPredictions('Mumbai', 'January');
      const risk = await rainfallAPI.getRiskAssessment('Mumbai');
      
      setRealData({ predictions, risk });
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <div>Loading dashboard...</div>;
  }

  // For simulated Data
  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     setLastUpdate(new Date().toLocaleString());
  //   }, 60000);
  //   return () => clearInterval(interval);
  // }, []);

    useEffect(() => {
    fetchData();
    }, []);

  // Sample data for visualizations
  const monthlyRainfall = [
    { month: 'Jan', actual: 45, predicted: 48, normal: 50 },
    { month: 'Feb', actual: 52, predicted: 55, normal: 55 },
    { month: 'Mar', actual: 78, predicted: 82, normal: 85 },
    { month: 'Apr', actual: 125, predicted: 130, normal: 140 },
    { month: 'May', actual: 180, predicted: 190, normal: 200 },
    { month: 'Jun', actual: 450, predicted: 480, normal: 500 },
    { month: 'Jul', actual: 680, predicted: 700, normal: 750 },
    { month: 'Aug', actual: 620, predicted: 640, normal: 700 },
    { month: 'Sep', actual: 340, predicted: 360, normal: 380 },
    { month: 'Oct', actual: 120, predicted: null, normal: 140 },
    { month: 'Nov', actual: 65, predicted: null, normal: 75 },
    { month: 'Dec', actual: 40, predicted: null, normal: 45 }
  ];

  const riskAssessment = [
    { metric: 'Drought Risk', current: 85, previous: 70 },
    { metric: 'Flood Risk', current: 45, previous: 50 },
    { metric: 'Crop Failure', current: 72, previous: 60 },
    { metric: 'Water Scarcity', current: 78, previous: 65 },
    { metric: 'Economic Impact', current: 68, previous: 55 },
    { metric: 'Food Security', current: 65, previous: 58 }
  ];

  const stateVulnerability = [
    { state: 'Maharashtra', index: 78, category: 'Critical' },
    { state: 'Karnataka', index: 72, category: 'Critical' },
    { state: 'Andhra Pradesh', index: 68, category: 'High' },
    { state: 'Tamil Nadu', index: 65, category: 'High' },
    { state: 'Gujarat', index: 82, category: 'Critical' },
    { state: 'Rajasthan', index: 88, category: 'Critical' },
    { state: 'Madhya Pradesh', index: 70, category: 'High' },
    { state: 'Uttar Pradesh', index: 66, category: 'High' },
    { state: 'Bihar', index: 74, category: 'Critical' },
    { state: 'West Bengal', index: 58, category: 'Moderate' },
    { state: 'Odisha', index: 62, category: 'High' },
    { state: 'Kerala', index: 45, category: 'Moderate' }
  ];

  const modelPerformance = [
    { model: 'LSTM Neural Network', accuracy: 91, rmse: 102, mae: 82 },
    { model: 'Random Forest', accuracy: 88, rmse: 112, mae: 89 },
    { model: 'Gradient Boosting', accuracy: 86, rmse: 118, mae: 94 },
    { model: 'ARIMA-GARCH', accuracy: 85, rmse: 124, mae: 98 },
    { model: 'Ensemble', accuracy: 92, rmse: 98, mae: 78 }
  ];

  const cropImpact = [
    { crop: 'Rice', impact: -15, area: 44000 },
    { crop: 'Wheat', impact: -8, area: 31000 },
    { crop: 'Cotton', impact: -22, area: 12500 },
    { crop: 'Sugarcane', impact: -18, area: 5000 },
    { crop: 'Pulses', impact: -12, area: 29000 },
    { crop: 'Oilseeds', impact: -10, area: 26000 }
  ];

  const economicImpact = [
    { category: 'Agricultural Loss', value: 3700, change: 18 },
    { category: 'Insurance Claims', value: 2100, change: 25 },
    { category: 'Relief Measures', value: 1500, change: 12 },
    { category: 'Infrastructure Damage', value: 800, change: -5 },
    { category: 'Recovery Investment', value: 2500, change: 30 }
  ];

  const getColorByRisk = (value) => {
    if (value > 75) return '#ef4444';
    if (value > 50) return '#f59e0b';
    if (value > 25) return '#10b981';
    return '#3b82f6';
  };

  const MetricCard = ({ icon: Icon, label, value, change, color = 'blue' }) => {
    const colorClasses = {
      blue: 'from-blue-500 to-blue-600',
      red: 'from-red-500 to-red-600',
      green: 'from-green-500 to-green-600',
      orange: 'from-orange-500 to-orange-600',
      purple: 'from-purple-500 to-purple-600'
    };

    return (
      <div className={`relative bg-gradient-to-br ${colorClasses[color]} rounded-xl p-6 text-white overflow-hidden transition-transform hover:scale-105`}>
        <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-white bg-opacity-10 rounded-full"></div>
        <div className="relative z-10">
          <Icon className="w-8 h-8 mb-2 opacity-80" />
          <div className="text-3xl font-bold mb-1">{value}</div>
          <div className="text-sm opacity-90">{label}</div>
          {change && (
            <div className="flex items-center mt-2 text-xs">
              {change > 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              <span>{Math.abs(change)}% from last year</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const refreshData = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      setLastUpdate(new Date().toLocaleString());
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Indian Rainfall Analytics Dashboard
              </h1>
              <p className="text-gray-600 mt-1">Government Decision Support System | Real-time Climate Intelligence</p>
            </div>
            <button
              onClick={refreshData}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
          
          {/* Alert Banner */}
          {alertVisible && (
            <div className="bg-gradient-to-r from-red-500 to-orange-500 text-white p-4 rounded-lg flex items-center justify-between animate-pulse">
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-6 h-6" />
                <div>
                  <strong>CRITICAL ALERT:</strong> Predicted 15% below-normal rainfall in Maharashtra for next 72 hours.
                  Activate drought contingency measures in 12 districts.
                </div>
              </div>
              <button onClick={() => setAlertVisible(false)} className="text-white hover:text-gray-200">×</button>
            </div>
          )}
        </div>

        {/* Control Panel */}
        <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
          <div className="flex flex-wrap gap-4">
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="px-4 py-2 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none"
            >
              <option>All India</option>
              <option>Northern Region</option>
              <option>Southern Region</option>
              <option>Western Region</option>
              <option>Eastern Region</option>
              <option>Northeast Region</option>
            </select>
            
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="px-4 py-2 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:outline-none"
            >
              <option>Current Year</option>
              <option>Last 30 Days</option>
              <option>Last 90 Days</option>
              <option>Current Monsoon</option>
              <option>5-Year Trend</option>
            </select>

            <div className="flex gap-2 ml-auto">
              {['overview', 'predictions', 'risk', 'impact'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-lg capitalize transition-colors ${
                    activeTab === tab
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <MetricCard
            icon={Cloud}
            label="Annual Rainfall 2024"
            value="1,087mm"
            change={-6.2}
            color="blue"
          />
          <MetricCard
            icon={Target}
            label="Prediction Accuracy"
            value="87%"
            change={3}
            color="green"
          />
          <MetricCard
            icon={AlertCircle}
            label="Districts at Risk"
            value="245"
            change={18}
            color="red"
          />
          <MetricCard
            icon={DollarSign}
            label="Estimated Impact"
            value="₹3.7B"
            change={12}
            color="orange"
          />
        </div>

        {/* Main Content Area */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Monthly Rainfall Trends */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">Monthly Rainfall Trends & Predictions</h3>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={monthlyRainfall}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="month" stroke="#666" />
                  <YAxis stroke="#666" />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px' }}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="normal" stackId="1" stroke="#10b981" fill="#10b98120" name="Historical Average" />
                  <Area type="monotone" dataKey="actual" stackId="2" stroke="#3b82f6" fill="#3b82f640" name="Actual 2024" />
                  <Area type="monotone" dataKey="predicted" stackId="3" stroke="#ef4444" fill="#ef444430" name="AI Prediction" strokeDasharray="5 5" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* State Vulnerability */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">State-wise Vulnerability Index</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={stateVulnerability} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis type="number" domain={[0, 100]} stroke="#666" />
                  <YAxis type="category" dataKey="state" stroke="#666" width={100} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px' }}
                  />
                  <Bar dataKey="index" name="Vulnerability Index">
                    {stateVulnerability.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getColorByRisk(entry.index)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'predictions' && (
          <div className="space-y-6">
            {/* AI Predictions Panel */}
            <div className="bg-gradient-to-r from-green-500 to-teal-600 text-white rounded-2xl shadow-xl p-6">
              <h3 className="text-2xl font-bold mb-4">🔮 AI-Powered Predictions (Next 30 Days)</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {[
                  { label: 'Monsoon Onset', value: 'Kerala: June 8 (±2 days)' },
                  { label: 'Extreme Events', value: '3 Heavy Rainfall Events' },
                  { label: 'Drought Risk', value: 'High: 42 Districts' },
                  { label: 'Flood Risk', value: 'Moderate: 18 Districts' },
                  { label: 'Confidence Level', value: '91% (Ensemble Model)' }
                ].map((item, idx) => (
                  <div key={idx} className="bg-white bg-opacity-20 backdrop-blur-lg rounded-lg p-4">
                    <div className="font-semibold mb-1">{item.label}</div>
                    <div className="text-sm">{item.value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Performance */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">Machine Learning Model Performance</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={modelPerformance}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="model" angle={-45} textAnchor="end" height={100} stroke="#666" />
                  <YAxis stroke="#666" />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px' }}
                  />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy (%)" />
                  <Bar dataKey="rmse" fill="#ef4444" name="RMSE (mm)" />
                  <Bar dataKey="mae" fill="#10b981" name="MAE (mm)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'risk' && (
          <div className="space-y-6">
            {/* Risk Assessment Radar */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">Multi-Dimensional Risk Assessment</h3>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={riskAssessment}>
                  <PolarGrid stroke="#e0e0e0" />
                  <PolarAngleAxis dataKey="metric" stroke="#666" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#666" />
                  <Radar name="Current Assessment" dataKey="current" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} />
                  <Radar name="Previous Year" dataKey="previous" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Risk Matrix */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-xl font-bold mb-4 text-gray-800">District Risk Categories</h3>
                <div className="space-y-3">
                  {[
                    { level: 'Critical', count: 42, color: 'red' },
                    { level: 'High', count: 78, color: 'orange' },
                    { level: 'Moderate', count: 95, color: 'yellow' },
                    { level: 'Low', count: 30, color: 'green' }
                  ].map((item) => (
                    <div key={item.level} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className={`w-4 h-4 rounded-full bg-${item.color}-500`}></div>
                        <span className="font-medium">{item.level} Risk</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-2xl font-bold">{item.count}</span>
                        <span className="text-gray-500">districts</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-xl p-6">
                <h3 className="text-xl font-bold mb-4 text-gray-800">Recommended Actions</h3>
                <div className="space-y-3">
                  {[
                    { action: 'Activate Emergency Response', priority: 'Immediate', icon: AlertTriangle },
                    { action: 'Deploy Water Tankers', priority: 'Within 24 hrs', icon: Droplets },
                    { action: 'Crop Insurance Processing', priority: 'Within 48 hrs', icon: DollarSign },
                    { action: 'Relief Distribution', priority: 'Within 72 hrs', icon: Users }
                  ].map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg hover:from-blue-100 hover:to-indigo-100 transition-colors cursor-pointer">
                      <div className="flex items-center gap-3">
                        <item.icon className="w-5 h-5 text-blue-600" />
                        <span className="font-medium">{item.action}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <span>{item.priority}</span>
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'impact' && (
          <div className="space-y-6">
            {/* Agricultural Impact */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">Agricultural Impact Analysis</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={cropImpact}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="crop" stroke="#666" />
                  <YAxis stroke="#666" />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px' }}
                    formatter={(value, name) => {
                      if (name === 'impact') return [`${value}%`, 'Yield Impact'];
                      if (name === 'area') return [`${value.toLocaleString()} hectares`, 'Affected Area'];
                      return [value, name];
                    }}
                  />
                  <Legend />
                  <Bar dataKey="impact" fill="#ef4444" name="Yield Impact (%)">
                    {cropImpact.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.impact < -15 ? '#ef4444' : entry.impact < -10 ? '#f59e0b' : '#10b981'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Economic Impact */}
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">Economic Impact Breakdown</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {economicImpact.map((item, idx) => (
                  <div key={idx} className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-4 hover:shadow-lg transition-shadow">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-700">{item.category}</h4>
                      <div className={`flex items-center text-sm ${item.change > 0 ? 'text-red-600' : 'text-green-600'}`}>
                        {item.change > 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                        <span>{Math.abs(item.change)}%</span>
                      </div>
                    </div>
                    <div className="text-2xl font-bold text-gray-900">₹{item.value}M</div>
                    <div className="text-sm text-gray-600 mt-1">Million INR</div>
                  </div>
                ))}
              </div>
              
              {/* Total Impact Summary */}
              <div className="mt-6 p-4 bg-gradient-to-r from-red-50 to-orange-50 rounded-lg border-l-4 border-red-500">
                <div className="flex justify-between items-center">
                  <div>
                    <h4 className="font-semibold text-gray-800">Total Economic Impact</h4>
                    <p className="text-sm text-gray-600 mt-1">Combined losses across all sectors</p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-red-600">₹10.6B</div>
                    <div className="text-sm text-gray-600">+20% from last year</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Policy Recommendations */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-xl p-6">
              <h3 className="text-xl font-bold mb-4 text-gray-800">📋 Policy Recommendations</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  {
                    title: 'Immediate Actions (0-3 months)',
                    items: [
                      'Activate drought contingency funds (₹500 Cr)',
                      'Fast-track crop insurance claims',
                      'Deploy mobile water tankers to 245 villages',
                      'Distribute drought-resistant seeds'
                    ],
                    color: 'red'
                  },
                  {
                    title: 'Short-term Measures (3-6 months)',
                    items: [
                      'Expand micro-irrigation to 50,000 hectares',
                      'Establish 100 rainwater harvesting structures',
                      'Launch farmer training programs',
                      'Implement crop diversification schemes'
                    ],
                    color: 'orange'
                  },
                  {
                    title: 'Medium-term Strategy (6-12 months)',
                    items: [
                      'Develop AI-based early warning system',
                      'Create district-level water budgets',
                      'Establish climate-smart villages',
                      'Build check dams in 50 watersheds'
                    ],
                    color: 'blue'
                  },
                  {
                    title: 'Long-term Vision (1-3 years)',
                    items: [
                      'Achieve 35% micro-irrigation coverage',
                      'Restore 50,000 sq km forest cover',
                      'Implement blockchain-based insurance',
                      'Create regional climate adaptation centers'
                    ],
                    color: 'green'
                  }
                ].map((section, idx) => (
                  <div key={idx} className="bg-white rounded-lg p-4">
                    <h4 className={`font-semibold text-${section.color}-600 mb-3`}>{section.title}</h4>
                    <ul className="space-y-2">
                      {section.items.map((item, itemIdx) => (
                        <li key={itemIdx} className="flex items-start gap-2 text-sm text-gray-700">
                          <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="bg-white rounded-2xl shadow-xl p-6 mt-6">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-600">
              <strong>Data Sources:</strong> IMD, ICAR, Satellite Data (TRMM/GPM) | 
              <strong className="ml-2">Model Accuracy:</strong> 87% | 
              <strong className="ml-2">Coverage:</strong> 640 Districts
            </div>
            <div className="text-sm text-gray-600">
              <strong>Last Updated:</strong> {lastUpdate}
            </div>
          </div>
          <div className="mt-4 flex gap-4">
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              <Download className="w-4 h-4" />
              Export Report
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors">
              <Activity className="w-4 h-4" />
              View Detailed Analytics
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors">
              <Map className="w-4 h-4" />
              Open GIS Portal
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RainfallDashboard;