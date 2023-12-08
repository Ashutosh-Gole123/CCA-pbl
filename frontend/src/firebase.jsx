// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
const firebaseConfig = {
  apiKey: "AIzaSyCIktjTsN5MGeL7Zh2e_orBqftFrLuk-kE",
  authDomain: "designproject-d0e7a.firebaseapp.com",
  projectId: "designproject-d0e7a",
  storageBucket: "designproject-d0e7a.appspot.com",
  messagingSenderId: "976889504508",
  appId: "1:976889504508:web:a70e8c8e9af247395fad79",
  measurementId: "G-PZL9FVSPBT",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();
export { app, auth, provider };
