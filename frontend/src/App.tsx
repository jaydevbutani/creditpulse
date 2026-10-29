import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { PortfolioDashboard } from './pages/PortfolioDashboard';
import { LoanDetail } from './pages/LoanDetail';
import { Login } from './pages/Login';
import { MyDashboard } from './pages/MyDashboard';
import { UploadCase } from './pages/UploadCase';
import { CaseHistory } from './pages/CaseHistory';
import { CaseDetail } from './pages/CaseDetail';
import { ModelHealth } from './pages/ModelHealth';
import { useAuth } from './context/AuthContext';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background text-outline text-xs">
        <span className="w-6 h-6 border-2 border-cognac border-t-transparent rounded-full animate-spin mr-2"></span>
        <span>Verifying institutional session...</span>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

export const App: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background font-sans text-slateInk">
      <Navbar />
      <main className="flex-1">
        <Routes>
          {/* Public Institutional Berka Monitor */}
          <Route path="/" element={<PortfolioDashboard />} />
          <Route path="/portfolio" element={<PortfolioDashboard />} />
          <Route path="/loan-detail" element={<LoanDetail />} />
          <Route path="/model-health" element={<ModelHealth />} />

          {/* Authentication */}
          <Route path="/login" element={<Login />} />

          {/* User Isolated Workspaces (Protected) */}
          <Route
            path="/my-dashboard"
            element={
              <ProtectedRoute>
                <MyDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/upload"
            element={
              <ProtectedRoute>
                <UploadCase />
              </ProtectedRoute>
            }
          />
          <Route
            path="/history"
            element={
              <ProtectedRoute>
                <CaseHistory />
              </ProtectedRoute>
            }
          />
          <Route
            path="/cases/:id"
            element={
              <ProtectedRoute>
                <CaseDetail />
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;
