<?php
	session_start();
	require_once("config.php");

	// get webdata
	$RAcode = $_POST["RPinputAcode"];
	$RPEmail = $_SESSION["Email"];
	
	// connect to db
	$con = mysqli_connect(SERVER,USER,PASSWORD,DATABASE);
	if (!$con) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "database connection failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// build query
	$query = "SELECT * FROM Users WHERE Email='$RPEmail' AND Acode = '$RAcode';";
	$result = mysqli_query($con,$query);
	if (!$result) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "authentication select query failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// query worked - doesn't mean authentication is successful - must check if only single item is returned
	if (mysqli_num_rows($result) != 1) {
		$_SESSION["RegState"] = -3;
		$_SESSION["Message"] = "reauthentication failed";
		header("location:../index.php");
		exit();
	}
	
	// switch view to resetPasswordForm
	$_SESSION["RegState"] = 3;
	$_SESSION["Message"] = "Re-authentication successful. Ready to reset password.";
	header("location:../index.php");
	exit();
?>