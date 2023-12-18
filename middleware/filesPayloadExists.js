const filesPayloadExists = (req, res, next) => {
    if (!req.files) return res.status(400).json({ status: "Error", message: "No file attached" })

    next()
}

module.exports = filesPayloadExists