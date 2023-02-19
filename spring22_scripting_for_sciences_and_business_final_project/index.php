<?php
	session_start();
	if (!isset($_SESSION["RegState"])) {
		$_SESSION["RegState"] = 0;
	}
?>
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="">
    <meta name="author" content="Nick McCloskey based on Mark Otto, Jacob Thornton, and Bootstrap contributors">
    <meta name="generator" content="Jekyll v3.8.5">
    <title>Lab11. Web Access of Local Applications</title>

    <!-- Bootstrap core CSS -->
	<link href="css/bootstrap.min.css" rel="stylesheet">

    <style>
      .bd-placeholder-img {
        font-size: 1.125rem;
        text-anchor: middle;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        user-select: none;
      }

      @media (min-width: 768px) {
        .bd-placeholder-img-lg {
          font-size: 3.5rem;
        }
      }
    </style>
    <!-- Custom styles for this template -->
    <link href="css/signin.css" rel="stylesheet">
	<script src="js/jquery-3.3.1.slim.min.js"></script>
	<script src="js/bootstrap.bundle.min.js"></script>
	<!-- reCAPTCHA widget -->
	<script src="https://www.google.com/recaptcha/api.js" async defer></script>
		<!--script>
			$(document).ready(function() {
				$("#loginForm").hide();
				$("#setPasswordForm").hide();
				$("#registrationProcessForm").show();
			})
		</script-->
  </head>
  <body class="text-center">
<?php
	if ($_SESSION["RegState"] <= 0) {
?>
	
	<form id="loginForm" class="form-signin" action="php/login.php" method="POST">
			<img class="mb-4" src="images/bootstrap-solid.svg" alt="" width="72" height="72">
			<h1 class="h3 mb-3 font-weight-normal">CIS5015 Project</h1>
			<label for="inputEmail" class="sr-only">Email address</label>
			<input type="email" name="inputEmail" class="form-control" placeholder="Email address" required autofocus>
			<label for="inputPassword" class="sr-only">Password</label>
			<input type="password" name="inputPassword" class="form-control" placeholder="Password" required>
			<div class="checkbox mb-3">
				<label>
					<input type="checkbox" name="rememberMe" value="remember-me" id="rememberMe"> Remember me
				</label>
			</div>
			<button class="btn btn-lg btn-primary btn-block" type="submit">Sign in</button>
			<div class="alert alert-info mt-2" id="MessageBox">
				<?php
					print $_SESSION["Message"];
					$_SESSION["Message"] = "";
				?>
			</div>
			<br>
			<a href="php/register0.php">Register</a> | <a href="php/forget.php">Forget?</a>
			<p class="mt-5 mb-3 text-muted">&copy; 2017-2019</p>
		</form>
	
<?php
	}
	if ($_SESSION["RegState"] == 2) {
?>
	<!--script>
			$(document).ready(function() {
				$("#loginForm").hide();
				$("#setPasswordForm").hide();
				$("#registrationProcessForm").show();
			})
	</script-->
	<div id="registrationProcessForm" class="form-signin">
		<form id="registerForm" class="form-signin" action="php/register.php" method="GET">
			<img class="mb-4" src="images/bootstrap-solid.svg" alt="" width="72" height="72">
			<h1 class="h3 mb-3 font-weight-normal">CIS5015 Project</h1>
			<label for="inputFirstName" class="sr-only">First Name</label>
			<input type="text" name="inputFirstName" id="inputFirstName" class="form-control" placeholder="First Name" required autofocus>
			<label for="inputLastName" class="sr-only">Last Name</label>
			<input type="text" name="inputLastName" id="inputLastName" class="form-control" placeholder="Last Name" required autofocus>
			<label for="inputEmail" class="sr-only">Email address</label>
			<input type="email" name="inputEmail" id="inputEmail" class="form-control" placeholder="Email address" required autofocus>
			<br>
			<!-- reCAPTCHA widget -->
			<div>
				enter the captcha code below
				<img id="captcha" src="securimage/securimage/securimage_show.php" alt="CAPTCHA Image" />
				<input type="text" name="captcha_code" id="captcha_code" size="10" maxlength="6" />
				<a href="#" onclick="document.getElementById('captcha').src = 'securimage/securimage/securimage_show.php?' + Math.random(); return false">[Different Image]</a>
			</div>
			<br>
			<button class="btn btn-lg btn-primary btn-block" type="submit">Register</button>
			<div class="alert alert-info mt-2" id="RMessageBox">
				<?php
					print $_SESSION["Message"];
					$_SESSION["Message"] = "";
				?>
			</div>
		</form>
				
		<form id="authenticationForm" class="form-signin" action="php/authenticate.php" method="POST">
			<label for="Acode" class="sr-only">Authentication Code</label>
			<input type="text" name="inputAcode" id="inputAcode" class="form-control" placeholder="authentication code from email" required autofocus>
			<button class="btn btn-lg btn-primary btn-block" type="submit">Authenticate</button>
			<div class="alert alert-info mt-2" id="MessageBox">
				<?php
					print $_SESSION["Message"];
					$_SESSION["Message"] = "";
				?>
			</div>
			<br>
			<a href="php/return.php">Return</a>
			<p class="mt-5 mb-3 text-muted">&copy; 2017-2019</p>
		</form>
	</div>	
<?php
	}
	if ($_SESSION["RegState"] == 3) {
?>
		<form id="setPasswordForm" class="form-signin" action="php/setPassword.php" method="POST">
			<img class="mb-4" src="images/bootstrap-solid.svg" alt="" width="72" height="72">
			<h1 class="h3 mb-3 font-weight-normal">CIS5015 Set Password</h1>	
			<label for="inputPassword" class="sr-only">Password</label>
			<input type="password" name="inputPassword" id="inputPassword" class="form-control" placeholder="Password" required autofocus>
			<label for="password2" class="sr-only">Password again</label>
			<input type="password" name="password2" id="password2" class="form-control" placeholder="Password again" required autofocus>
			Password Strength:
			<script>
				// from http://jsfiddle.net/HFMvX/
				function scorePassword(pass) {
					var score = 0;
					if (!pass)
						return score;

					// award every unique letter until 5 repetitions
					var letters = new Object();
					for (var i=0; i<pass.length; i++) {
						letters[pass[i]] = (letters[pass[i]] || 0) + 1;
						score += 5.0 / letters[pass[i]];
					}

					// bonus points for mixing it up
					var variations = {
						digits: /\d/.test(pass),
						lower: /[a-z]/.test(pass),
						upper: /[A-Z]/.test(pass),
						nonWords: /\W/.test(pass),
					}

					variationCount = 0;
					for (var check in variations) {
						variationCount += (variations[check] == true) ? 1 : 0;
					}
					score += (variationCount - 1) * 10;

					return parseInt(score);
				}

				function checkPassStrength(pass) {
					var score = scorePassword(pass);
					if (score > 80)
						return "strong";
					if (score > 60)
						return "good";
					if (score >= 30)
						return "weak";
					if (score < 30)
						return "very weak";

					return "";
				}
				
				$(document).ready(function() {
					$("#inputPassword").on("keypress keyup keydown", function() {
						var pass = $(this).val();
						$("#passwordStrength").text(checkPassStrength(pass));
					});
				});
			</script>
			<div class="figure" id="passwordStrength">start typing</div>
			<button class="btn btn-lg btn-primary btn-block mt-2" type="submit">Set Password</button>
			<div class="alert alert-info mt-2" id="SPMessageBox">
				<?php
					print $_SESSION["Message"];
					$_SESSION["Message"] = "";
				?>
			</div>
			<br>
			<a href="php/return.php">Return</a>
			<p class="mt-5 mb-3 text-muted">&copy; 2017-2019</p>
		</form>

<?php
	}
	if ($_SESSION["RegState"] == 6) {
?>
		<div id="resetPasswordForm" class="form-signin">
			<form id="resendAcodeForm" class="form-signin" action="php/resetPassword.php" method="GET">
				<img class="mb-4" src="images/bootstrap-solid.svg" alt="" width="72" height="72">
				<h1 class="h3 mb-3 font-weight-normal">CIS5015 Project Reset Password</h1>
				<label for="RPinputEmail" class="sr-only">Email address</label>
				<input type="email" name="RPinputEmail" id="RPinputEmail" class="form-control" placeholder="Email address" required autofocus>
				<button class="btn btn-lg btn-primary btn-block" type="submit">Send Authentication Code</button>
				<div class="alert alert-info mt-2" id="RMessageBox">
					<?php
						print $_SESSION["Message"];
						$_SESSION["Message"] = "";
					?>
				</div>
				<br>
			</form>	
		
			<form id="RPauthenticationForm" class="form-signin" action="php/reauthenticate.php" method="POST">
				<label for="Acode" class="sr-only">Authentication Code</label>
				<input type="text" name="RPinputAcode" id="RPinputAcode" class="form-control" placeholder="authentication code from email" required autofocus>
				<button class="btn btn-lg btn-primary btn-block" type="submit">Authenticate</button>
				<div class="alert alert-info mt-2" id="MessageBox">
					<?php
						print $_SESSION["Message"];
						$_SESSION["Message"] = "";
					?>
				</div>
				<br>
				<a href="php/return.php">Return</a>
				<p class="mt-5 mb-3 text-muted">&copy; 2017-2019</p>
			</form>
		</div>
<?php
	}
	exit();
?>
	</body>
</html>