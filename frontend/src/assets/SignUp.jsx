import { useNavigate } from "react-router-dom";
import { useUserAuth } from "../context/UserAuthContext";
import { Box, Button, Divider, TextField } from "@mui/material";
import GoogleIcon from "@mui/icons-material/Google";
import img from "./images/signup.jpg";
import { Reveal } from "./animates/Reveal";
function SignUp() {
  const { googleSignIn } = useUserAuth();
  const navigate = useNavigate();
  const handleGoogleSignIn = async (e) => {
    e.preventDefault();
    try {
      await googleSignIn();
      navigate("/home");
    } catch (error) {
      console.log(error.message);
    }
  };
  return (
    <div
      style={{
        backgroundImage: `url(${img})`,
        height: "100vh",
        backgroundSize: "cover",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          background: "rgba(0, 0, 0, 0.6)", // Adjust the opacity (0.5) and color if needed
          zIndex: 1, // Ensure the overlay is above the background image
        }}
      >
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            flexWrap: "wrap",
            alignContent: "center",
            justifyContent: "center",
            height: "100vh",
            width: "100vh",
          }}
        >
          <Box
            style={{
              display: "flex",
              flexDirection: "column",
              margin: "10px",
              color: "white",
            }}
          >
            <Reveal>
              <TextField
                label="Email"
                color="primary"
                InputProps={{
                  style: { color: "white" },
                }}
                focused
                sx={{ marginTop: "10px", marginBottom: "10px" }}
              />
            </Reveal>
            <Reveal>
              <TextField
                label="Password"
                color="primary"
                focused
                InputProps={{
                  style: { color: "white" },
                }}
                sx={{ marginTop: "10px", marginBottom: "10px" }}
              />
            </Reveal>
            <Reveal>
              <Button fullWidth variant="contained">
                SIGN IN
              </Button>
            </Reveal>
          </Box>
          <Divider sx={{ color: "white", marginBottom: "10px" }}>OR</Divider>

          <Reveal>
            <Button
              fullWidth
              variant="contained"
              onClick={handleGoogleSignIn}
              style={{ backgroundColor: "white", color: "#333" }}
            >
              <GoogleIcon style={{ color: "#4AABE8", marginRight: "10px" }} />{" "}
              SIGN UP
            </Button>
          </Reveal>
        </Box>
      </div>
    </div>
  );
}

export default SignUp;
