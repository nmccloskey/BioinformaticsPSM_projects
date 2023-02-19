<?php
	session_start();
	if ($_SESSION["RegState"] != 4) {
		$_SESSION["RegState"] = 0;
		$_SESSION["Message"] = "Please login first";
		header("location:index.php");
		exit();
	}
?>
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="CIS5015 Lab7">
    <meta name="author" content="Nick McCloskey based on Mark Otto, Jacob Thornton, and Bootstrap contributors">
    <meta name="generator" content="Jekyll v3.8.5">
    <title>Lab11. Web Access of Local Applications</title>
    <link rel="icon" type="image/x-icon" href="images/favicon.ico">
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
    <link href="css/dashboard.css" rel="stylesheet">
	<script src="js/jquery-3.6.0.min.js"></script>
	<script src="js/bootstrap.bundle.min.js"></script>
	<script src="js/feather.min.js"></script>
	<!-- <script src="js/Chart.min.js"></script> -->
	<!-- <script src="js/dashboard.js"></script> -->
	<script src="js/lab11.js"></script>
	<script src="js/fa.js"></script>
  </head>
  
  <body>
    <nav class="navbar navbar-dark fixed-top bg-dark flex-md-nowrap navbar-expand-md p-0 shadow">
	  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#collapsibleNavbar">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="collapsibleNavbar">
	    <a class="navbar-brand col-sm-3 col-md-2 mr-0" href="#">tuf61393</a>
        <input class="form-control form-control-dark w-100" type="text" placeholder="Search" aria-label="Search">
          <ul class="navbar-nav px-3">
            <li class="nav-item text-nowrap">
              <a class="nav-link text-nowrap ml-2" href="php/logout.php">Sign out</a>
            </li>
          </ul>
				</input>
      </div>
    </nav>

	<div class="container-fluid">
	  <div class="row">
		<nav class="col-md-2 d-none d-md-block bg-light sidebar">
		  <div class="sidebar-sticky">
			<ul class="nav flex-column">
			  <li class="nav-item" id="showAll">
				<a class="nav-link active" href="#">
				  <span id="showAllText"><i class="fas fa-home"></i></span>
				  Hide Dashboard<span class="sr-only">(current)</span>
				</a>
			  </li>
			  <li class="nav-item" id="run61">
				<a class="nav-link" href="#">
				  <i class="fas fa-dot-circle"></i>
				  Hide Run Lab6.1
				</a>
			  </li>
			  <li class="nav-item" id="run62">
				<a class="nav-link" href="#">
				  <i class="fas fa-dot-circle"></i>
				  Hide Run Lab6.2
				</a>
			  </li>
			  <li class="nav-item" id="report">
				<a class="nav-link" href="#">
				  <i class="fa-solid fa-lock"></i>
				  Hide Report
				</a>
			  </li>
			</ul>
		  </div>
		</nav>

		<main role="main" class="col-md-9 ml-sm-auto col-lg-10 px-4 pb-5">
		  <div class="d-none d-sm-block">
		    <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
			  <h1 class="h2">CIS5015: Web Service Scripting</h1>
			    <div class="btn-toolbar mb-2 mb-md-0">
			      <div class="btn-group mr-2">
				    <button type="button" class="btn btn-sm btn-outline-secondary">Share</button>
				    <button type="button" class="btn btn-sm btn-outline-secondary">Export</button>
			      </div>
			      <button type="button" class="btn btn-sm btn-outline-secondary dropdown-toggle">
				    <span data-feather="calendar"></span>
				    This week
			      </button>
			    </div>
		    </div>
		  </div>
		  
			<div id="Lab62">
				<h2>Lab6.2 Online</h2>
				<div class="card" id="card62">
				  <h5 class="card-header">Lab6.2 Locality Mystery</h5>
						<div class="card-body">
							<ul class="list-group">
								<li class="list-group-item d-flex justify-content-between align-items-center" id="lab62b1">
									<h2 class="btn btn-primary btn-small btn-block" data-toggle="collapse" >Loop Order Correctness Investigation</h2>
								</li>
								<div class="collapse in mt-3" id="card62_part1">
									<form>
										<label for="quantity">Size(3-20):</label>
										<!-- id was quantity-->
										<input type="number" id="matrixSize" name="matrixSize" min="3" max="20" value="3" required>
										<button class="btn btn-primary" id="lab62a" name="lab62a" type="submit">Run</button>
										<button class="btn btn-info" id="cleanLab621">
											Clear
										</button>
									</form>
									<hr>
									<div id="lab62_part1_runtime_monitor" name="lab62_part1_runtime_monitor" class="dropdown-item card-body">
										Results:
									</div>
								</div>
								<li class="list-group-item d-flex justify-content-between align-items-center" id="lab62b2">
									<h2 class="btn btn-primary btn-small btn-block" data-toggle="collapse" href="#card62_part2">Best Performing Order Investigation</h2>
								</li>
								<div class="collapse in" id="card62_part2">
									<div class="card card-body">
										<form>
											<label for="quantity">Size(100-2000):</label>
											<input type="number" id="matrixSize2" name="matrixSize2" min="100" max="2000" value="100" required>
											Repeat(2-5):
											<input type="number" id="mrepeat" name="mrepeat" min="2" max="5" value="2" required>
											<button class="btn btn-primary" id="lab62b" name="lab62b" type="submit">
												Run
											</button>
											<button class="btn btn-info" id="cleanLab622">
											Clear
											</button>
										</form>
										<hr>
										<div id="lab62_part2_loop_order_monitor" id="lab62_part2_loop_order_monitor" class="dropdown-item card-body">
											Results:
										</div>
										<hr>
										<button id="lab62b_plot" class="btn btn-primary btn-block">Plot</button>
										<canvas class="my-4" id="lab62chart" width="900" height="380"></canvas>
									</div>
								</div>
								<li class="list-group-item d-flex justify-content-between align-items-center" id="lab62b3">
									<span class="btn btn-primary btn-block" data-toggle="collapse" href="#card62_part3">Best Order Scalability Investigation</span>
								</li>
								<div class="collapse" id="card62_part3">
									<div class="card card-body">
										<form class="form-signin align-items-center">
											Loop Order:
											<input type="text" list="loop_orders" class="mb-2" name="order623" id="order623">
											<datalist id="loop_orders">
												<option value="ijk">
												<option value="ikj">
												<option value="jik">
												<option value="jki">
												<option value="kij">
												<option value="kji">
											</datalist>
											<div class="row">
												<input type="number" name="startN" id="startN" min="100" max="2000" class="col form-control" placeholder="Start(N): 100-800" required autofocus>
												<input type="number" name="stopN" min="100" max="800" class="col form-control" placeholder="End(N): 100-800">
												<input type="number" name="stepN" min="50" max="100" class="col form-control" placeholder="Step: 50-100">
											</div>
										<button class="btn btn-primary mt-3" id="lab62c" name="lab62c" type="submit">
											Run
										</button>
                                        <button class="btn btn-info mt-3" id="cleanLab623">
                                        Clear
                                        </button>
                                        </form>
                                        <hr>
                                            <div id="lab62_part3_scalability_monitor" id="lab62_part3_scalability_monitor" class="dropdown-item card-body">
                                                Results:
                                            </div>
										<hr>
										<button id="lab62_part3_plot" class="btn btn-primary btn-block">Plot</button>
										<canvas class="my-4" id="myChart2" width="900" height="380"></canvas>
									</div>
								</div>
							</ul>
						</div>
			  </div>
		  </div>
			
			<h2></h2>
			
		  <div id="Lab61">
				<h2>Lab6.1 Online</h2>
				<div class="card">
					<h5 class="card-header">Lab6.1 Sort Magic</h5>
					<div class="card-body">
						<h5 class="card-title">Bubble Sort Performance Comparisons</h5>
						<ul class="list-group list-group-flush">
							<li class="list-group-item" id="lab61b1">
								<h2 class="btn btn-primary btn-block" data-toggle="collapse" href="#lab61">Magic of Bubble Sort</h2>
							</li>
							<div class="collapse in" id="lab61">
								<form>
									<p class="card-text mt-2">This experiment demonstrates that partitioned input data can yield expected performance.</p>
									<label for="quantity">Quantity(1000-50000):</label>
									<input type="number" id="sortSize" name="sortSize" min="1000" max="50000" value="1000">
									<label for="repetition">Repeat(1-5):</label>
									<input type="number" id="repetition" name="repetition" min="1" max="5" value="1">
									<button class="btn btn-primary" id="runlab61" name="runlab61" type="submit">Run</button>
                                    <button class="btn btn-info" id="cleanLab61">
                                        Clear
                                    </button>
                                    <hr>
                                    <div id="lab61_monitor" id="lab61_monitor" class="dropdown-item card-body">
                                        Results:
                                    </div>
                                    <hr>
									<canvas class="my-4 w-100" id="sortChart" width="900" height="380"></canvas>
									<button id="sortPerfChart" class="btn btn-primary" type="submit">Plot</button>
								</form>
							</div>
						</ul>
					</div>
				</div>
		  </div>
			
			<h2></h2>
			
			<div id="Report">
				<h2>Lab6 Report</h2>
				<div class="card">
				<h5 class="card-header">Report</h5>
					<div class="card-body">
						<h5 class="card-title">Project Report</h5>
						<div id="lab6rb1">
						<h2 class="btn btn-primary btn-block" data-toggle="collapse" href="#projectReport">Details</h2>
						</div>
						<p id="projectReport" class="collapse in">This is the report.
						</p>
					</div>
				</div>
			</div>
		
		</main>
		
	  </div>
	</div>

  </body>
</html>
