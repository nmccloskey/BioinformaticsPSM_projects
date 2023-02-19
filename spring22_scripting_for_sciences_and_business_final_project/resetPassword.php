<?php
	session_start();
	require_once("config.php");

	use PHPMailer\PHPMailer\PHPMailer;
	use PHPMailer\PHPMailer\Exception;
	use PHPMailer\PHPMailer\SMTP;
	
	require "../../PHPMailer-master/PHPMailer-master/src/Exception.php";
	require "../../PHPMailer-master/PHPMailer-master/src/PHPMailer.php";
	require "../../PHPMailer-master/PHPMailer-master/src/SMTP.php";
	
	// get email
	$RPemail = $_GET["RPinputEmail"];
	$_SESSION["Email"] = $RPemail;
	
	// connect to database
	$con = mysqli_connect(SERVER,USER,PASSWORD,DATABASE);
	if (!$con) {
		$_SESSION["RegState"] = -1;
		$_SESSION["Message"] = "Database connection failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// generate new Acode
	$RAcode = rand();
	
	// create update query
	$query = "Update Users set Acode='$RAcode' where Email='$RPemail';";
	
	$result = mysqli_query($con, $query);
	if (!$result) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "Re-authentication insert failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	
	// create select query
	$query = "select * from Users where Email='$RPemail';";
	$result = mysqli_query($con, $query);
	
	$rows = mysqli_num_rows($result);
	if ($rows != 1) {
		$_SESSION["RegState"] = 0;
		$_SESSION["Message"] = "Error: email not found in database. Click below to register or try again.";
		header("location:../index.php");
		exit();
	}
	
	// send new authentication email
	$mail= new PHPMailer(true);
	try { 
		$mail->SMTPDebug = 2; // Wants to see all errors
		$mail->IsSMTP();
		$mail->Host="smtp.gmail.com";
		$mail->SMTPAuth=true;
		$mail->Username="indolentwallaby@gmail.com";
		$mail->Password = "55489532594852254";
		$mail->SMTPSecure = "ssl";
		$mail->Port=465;
		$mail->SMTPKeepAlive = true;
		$mail->Mailer = "smtp";
		$mail->setFrom("tuf61393@temple.edu", "Nick McCloskey");
		$mail->addReplyTo("tuf61393@temple.edu","Nick McCloskey");
		$msg = "Please enter code to re-authenticate: '$RAcode'";
		$mail->addAddress($RPemail,"$FirstName $LastName");
		$mail->Subject = "Welcome";
		$mail->Body = $msg;
		$mail->send();
		$_SESSION["RegState"] = 6;
		$_SESSION["Message"] = "Email sent.";
		header("location:../index.php");
		exit();
	} catch (phpmailerException $e) {
		$_SESSION["Message"] = "Mailer error: ".$e->errorMessage(); 
		$_SESSION["RegState"] = -4;
		print "Mail send failed: ".$e->errorMessage;
	}
	$_SESSION["RegState"] = 6;
	header("location:../index.php");
	exit();
	
	
	/*
	$RPassword = md5($_POST["inputresetPassword"]);
	$RPassword2 = md5($_POST["resetpassword2"]);
		
	// check for match
	if ($RPassword != $RPassword2) {
		$_SESSION["Message"] = "Passwords do not match. Enter passwords again.";
		header("location:../index.php");
		exit();
	}
	
	// connect to database
	$con = mysqli_connect(SERVER,USER,PASSWORD,DATABASE);
		if (!$con) {
		$_SESSION["RegState"] = -1;
		$_SESSION["Message"] = "Connection failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// create update query
	$query = "Update Users set Password='$RPassword' where Email='".$_SESSION["Email"]."';";
	$result = mysqli_query($con, $query);
	if (!result) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "Password reset failure: ".mysqli_error;
		header("location:../index.php");
		exit();
	}
	$_SESSION["RegState"] = 0;
	$_SESSION["Message"] = "Password reset. Please login.";
	header("location:../index.php");
	exit();*/
?>