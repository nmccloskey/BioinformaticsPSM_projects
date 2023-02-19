<?php
	header('Access-Control-Allow-Origin: *');
	session_start();
	$sortSize = $_GET["sortSize"];
	$repetition = $_GET["repetition"];
	//print "Size=($matrixSize) <br>";
	$command = escapeshellcmd("python/lab61.py ".$sortSize." ".$repetition." 2>&1");
	// print "command string: ($command)<br>";
	$output = shell_exec($command);
	echo $output; // direct text, not json format
?>