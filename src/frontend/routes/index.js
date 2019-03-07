var express = require('express');
var router = express.Router();
var _ = require('lodash');
var path = require('path');
var multer = require('multer');
var CXRStorage = require('../helpers/CXRStorage');
var request = require('request');

var storage = CXRStorage();
var limits = {
	files: 1,
	fileSize: 1024*1024,
};

var fileFilter = function(req, file, cb) {
	var allowedMimes = ['image/jpeg', 'image/pjpeg', 'image/png', 'image/gif'];

	if (_.includes(allowedMimes, file.mimetype)) {
		cb(null, true);
	} else {
		cb(new Error('Invalid file type. Only jpg, png and gif image files are allowed.'));
	}
};

var upload = multer({
	storage: storage,
	limits: limits,
	fileFilter: fileFilter
})

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Chest X-ray', cxr_field: 'cxr'});
});

router.post('/upload', upload.single('cxr'), function(req, res, next) {
	var file = req.file.filename;

	options = {
		uri: 'http://127.0.0.1:5002/cxr',
		method: 'POST',
		json: {
			image_name: file
		}
	}


	request(options, (err, result, body) => {
		if (err) { return console.log(err); }
		res.render('upload', {title: 'Result', body: body})
	})

});

module.exports = router;
