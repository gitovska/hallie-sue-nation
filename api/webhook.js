const crypto = require("crypto");

function createCrcResponseToken(crcToken) {
  const hmac = crypto
    .createHmac("sha256", process.env.TWITTER_CONSUMER_SECRET)
    .update(crcToken)
    .digest("base64");

  return `sha256=${hmac}`;
}

function getHandler(req, res) {
  const crcToken = req.query.crc_token;

  if (crcToken) {
    res.status(200).send({
      response_token: createCrcResponseToken(crcToken)
    });
  } else {
    res.status(400).send({
      message: "Error: crc_token missing from request."
    });
  }
}

function postHandler(req, res) {
  const body = req.body;
  console.log(body);
  res.status(200).json(body);
}

module.exports = (req, res) => {
  try {
    switch (req.method) {
      case "GET":
        return getHandler(req, res);
      case "POST":
        return postHandler(req, res);
      default:
        return res.status(410).json({ message: "Unsupported Request Method" });
    }
  } catch (error) {
    console.log(error.message);
    res.status(500).send();
  }
};

