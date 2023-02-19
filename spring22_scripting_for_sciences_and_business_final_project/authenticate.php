<?php
	// from old video
	session_start();
	require_once("config.php");
	
	// get webdata
	$Acode = $_POST["inputAcode"];
	//$Email = $_GET["inputEmail"];
	$Email = $_SESSION["Email"];
	
	// connect to db
	$con = mysqli_connect(SERVER,USER,PASSWORD,DATABASE);
	if (!$con) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "database connection failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// build query
	$query = "SELECT * FROM Users WHERE Email='$Email' AND Acode = '$Acode';";
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
		$_SESSION["Message"] = "authentication failed";
		header("location:../index.php");
		exit();
	}
	
	// reconnect to database
	$con = mysqli_connect(SERVER,USER,PASSWORD,DATABASE);
	if (!$con) {
		$_SESSION["RegState"] = -1;
		$_SESSION["Message"] = "Database connection failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// update database with Adatetime
	$Adatetime = date("Y-m-d h:i:s");
	$query = "update Users set Adatetime = '$Adatetime' where Email='$Email';";
	$result = mysqli_query($con, $query);
	if (!$result) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "Registration insert failed: $Adatetime and $Email !".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// switch view to setPasswordForm
	$_SESSION["RegState"] = 3;
	$_SESSION["Message"] = "Authentication successful. Ready to set password.";
	header("location:../index.php");
	exit();
	
	/*
	
	//from ppt
	$query="SELECT * FROM Users WHERE Email='$Email' and Acode='$Acode'";
	$status = mysqli_query($con,$query);
	if ($status) {
		$rows = mysqli_num_rows($status);
		if ($rows == 1) { //exact match. authenticated. ready to set password
			$_SESSION["RegState"] = 3;
			$_SESSION["Message"] = "Authentication successful. Please set password";
			header("location:../index.php");
			exit();
		} else {
			$_SESSION["RegState"] = -3;
			$_SESSION["Message"] = "Email Authentication Failed";
			header("location: ../index.php");
			exit();
		}
	}
	$_SESSION["RegState"] = -4;
	$_SESSION["Message"] = "Authentication query failure: ".mysqli_error($con);
	header("location:../index.php");
	exit();*/
/*
//from video
	print "DB connected <br>";
	// build the 'select'
	$query = "Select * from Users where Email='$Email' and Acode='$Acode';";
	$result = mysqli_query($con, $query);
	if (!result) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "Authentication select query failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	print "query worked <br>";
	
	$rows = mysqli_num_rows($result);
	if ($rows != 1) {
		$_SESSION["RegState"] = 3;
		$_SESSION["Message"] = "Either email or acode not match.";
		header("location:../index.php");
		exit();
	}
	print "Authentication successful <br>";
	$_SESSION["RegState"] = 3;
	$_SESSION["Email"] = $Email;
	$_SESSION["Message"] = "Authentication successful. Please set password.";
	header("location:../index.php");
	exit();*/
?>