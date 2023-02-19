<?php
	session_start();
	require_once("config.php");
	require_once("../securimage/securimage/securimage.php");
	
	use PHPMailer\PHPMailer\PHPMailer;
	use PHPMailer\PHPMailer\Exception;
	use PHPMailer\PHPMailer\SMTP;
	
	require "../../PHPMailer-master/PHPMailer-master/src/Exception.php";
	require "../../PHPMailer-master/PHPMailer-master/src/PHPMailer.php";
	require "../../PHPMailer-master/PHPMailer-master/src/SMTP.php";
	
	// check captcha form
	// they recommend this come after the other processes of this form, but it works best here
	// https://www.phpcaptcha.org/documentation/quickstart-guide/
	include_once $_SERVER['DOCUMENT_ROOT'] . 'securimage/securimage/securimage.php';
	$securimage = new Securimage();
	if ($securimage->check($_GET["captcha_code"]) == false) {
		// the code was incorrect
		// you should handle the error so that the form processor doesn't continue
		// or you can use the following code if there is no validation or you do not know how
		//echo "The security code entered was incorrect.<br /><br />";
		//echo "Please go <a href='javascript:history.go(-1)'>back</a> and try again.";
		$_SESSION["RegState"] = 2;
		$_SESSION["Message"] = "Recaptcha failed";
		header("location:../index.php");
		exit();
	}
	
	// get web data
	$FirstName = $_GET["inputFirstName"];
	$LastName = $_GET["inputLastName"];
	$Email = $_GET["inputEmail"];
	
	// save email for setpassword
	$_SESSION["Email"] = $Email;
	
	// connect to database
	$con = mysqli_connect(SERVER,USER,PASSWORD,DATABASE);
	if (!$con) {
		$_SESSION["RegState"] = -1;
		$_SESSION["Message"] = "Database connection failed: ".mysqli_error($con);
		header("location:../index.php");
		exit();
	}
	
	// build insert query
	$Acode = rand();
	$Rdatetime = date("Y-m-d h:i:s");
	$query = "Insert into Users (FirstName, LastName, Email, Acode, Rdatetime) values ('$FirstName', '$LastName', '$Email', '$Acode', '$Rdatetime');";
	$result = mysqli_query($con, $query);
	if (!$result) {
		$_SESSION["RegState"] = -2;
		$_SESSION["Message"] = "Registration insert failed: ".mysqli_error($con)." Try logging in.";
		header("location:../index.php");
		exit();
	}
	$_SESSION["RegState"] = 1;
	
	// Build the PHPMailer object:
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
		$msg = "Please enter code to authenticate: '$Acode'";
		$mail->addAddress($Email,"$FirstName $LastName");
		$mail->Subject = "Welcome";
		$mail->Body = $msg;
		$mail->send();
		$_SESSION["RegState"] = 2;
		$_SESSION["Message"] = "Email sent.";
		header("location:../index.php");
		exit();
	} catch (phpmailerException $e) {
		$_SESSION["Message"] = "Mailer error: ".$e->errorMessage(); 
		$_SESSION["RegState"] = -4;
		print "Mail send failed: ".$e->errorMessage;
	}
	$_SESSION["RegState"] = 2;
	header("location:../index.php");
	exit();	
?>