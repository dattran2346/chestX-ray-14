var _ = require('lodash');
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var Jimp = require('jimp');
var mkdirp = require('mkdirp');
var concat = require('concat-stream');
var streamifier = require('streamifier');

var UPLOAD_PATH = path.resolve(__dirname, '..', 'public/images/uploads');

var CXRStorage = function() {

	function CXRStorage() {

		var baseUrl = '/uploads';

		var allowedStorageSystems = ['local'];
		var allowedOutputFormats = ['jpg', 'png'];

		this.options = {
			// storage: 'local',
			output: 'png',
			greyscale: false,
			quality: 100,
			// square: true,
			// threshold: 1024,
			// responsive: false,
			};
		this.uploadPath = UPLOAD_PATH;
		this.uploadBaseUrl = '/uploads';
		!fs.existsSync(this.uploadPath) && mkdirp.sync(this.uploadPath);
	}

	CXRStorage.prototype._generateRandomFilename = function() {
		var bytes = crypto.pseudoRandomBytes(32);
		var checksum = crypto.createHash('MD5').update(bytes).digest('hex');
		return checksum + '.' + this.options.output;
	}

	CXRStorage.prototype._createOutputStream = function(filepath, cb) {
		var that = this;
		var output = fs.createWriteStream(filepath);
		output.on('error', cb);
		output.on('finish', function() {
			cb(null, {
				destination: that.uploadPath,
				baseUrl: that.uploadBaseUrl,
				filename: path.basename(filepath),
				// storage: that.opt
			})
		})
		return output;
	}

	CXRStorage.prototype._processImage = function(image, cb) {
		// process iamge and save upload file
		var that = this;
		var filename = this._generateRandomFilename();
		var mime = Jimp.MIME_PNG;
		var filepath = path.join(this.uploadPath, filename);
		outputStream = this._createOutputStream(filepath, cb);
		image.getBuffer(mime, function(err, buffer) {
			streamifier.createReadStream(buffer).pipe(outputStream);
		})
	}

	CXRStorage.prototype._handleFile = function(req, file, cb) {
		// handle upload file
		var that = this;
		var fileManipulate = concat(function(imageData) {
			Jimp.read(imageData)
				.then(function(image) {
					that._processImage(image, cb);
				})
				.catch(cb);
		});
		file.stream.pipe(fileManipulate);

	}

	CXRStorage.prototype._removeFile = function(req, file, cb) {
		var matches, pathsplit;
		var filename = file.filename;
		var _path = path.join(this.uploadPath, filename);

		delete file.filename;
		delete file.destination;
		delete file.baseUrl;
		delete file.storage;

		fs.unlink(_path, cb);
	}

	return new CXRStorage();
};

module.exports = CXRStorage;
