var express = require('express');
var router = express.Router();

router.get('/', function(req, res, next) {
	body = {
		image_name: 'd6f0d529d506433e8b99c0b9e5bee5fa.png',
		result: {
			'Infiltration': '0.425',
			'Atelectasis': '0.0758',
			'Nodule': '0.0645',
			'Pneumonia': '0.0518',
			'Fibrosis': '0.0387',
			'Pleural Thickening': '0.0386',
			'Edema': '0.0166',
			'Consolidation': '0.0149',
			'Cardiomegaly': '0.0126',
			'Mass': '0.0113'
		}
	}
  res.render('upload', { title: 'Result', body: body});
});

module.exports = router;
