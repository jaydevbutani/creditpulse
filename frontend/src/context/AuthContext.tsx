import React, { createContext, useContext, useState, useEffect } from 'react';
import { User } from '../types';
import { api } from '../api';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  login: (username: string, password?: string) => Promise<void>;
  loginAsRole: (role: 'analyst' | 'examiner') => Promise<void>;
  register: (payload: { username: string; email: string; password: string; full_name: string; role?: string }) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('cp_token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('cp_token');
      if (storedToken) {
        try {
          const me = await api.getMe();
          setUser(me);
          setToken(storedToken);
        } catch {
          localStorage.removeItem('cp_token');
          setUser(null);
          setToken(null);
        }
      }
      setLoading(false);
    };
    initAuth();
  }, []);

  const login = async (username: string, password?: string) => {
    const res = await api.login(username, password);
    localStorage.setItem('cp_token', res.token);
    setToken(res.token);
    setUser(res.user);
  };

  const loginAsRole = async (role: 'analyst' | 'examiner') => {
    await login(role);
  };

  const register = async (payload: { username: string; email: string; password: string; full_name: string; role?: string }) => {
    const res = await api.register(payload);
    localStorage.setItem('cp_token', res.token);
    setToken(res.token);
    setUser(res.user);
  };

  const logout = () => {
    localStorage.removeItem('cp_token');
    setUser(null);
    setToken(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        loading,
        login,
        loginAsRole,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
