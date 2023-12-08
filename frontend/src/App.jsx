import "./App.css";
import { Route, Routes } from "react-router-dom";
import Home from "./assets/Home";

import { UserAuthContextProvider } from "./context/UserAuthContext";
import SignUp from "./assets/SignUp";
import ProtectedRoute from "./assets/ProtectedRoute";

function App() {
  return (
    <>
      <div>
        <UserAuthContextProvider>
          <Routes>
            <Route
              path="/home"
              element={
                <ProtectedRoute>
                  <Home />
                </ProtectedRoute>
              }
            />
            <Route path="/" element={<SignUp />} />
          </Routes>
        </UserAuthContextProvider>
      </div>
    </>
  );
}

export default App;
