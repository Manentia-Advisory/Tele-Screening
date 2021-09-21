const Remark = (props) => (
    <div className="relative mx-12 mb-10">
        <h3 className="font-medium text-gray-900">Remarks</h3>
            <div className="border border-gray-300 rounded">
                <div className="m-1 max-w-full flex items-center">
                    <textarea 
                        className="w-full h-20 sm:h-28 resize-y break-words border-0 outline-none focus:outline-none focus:ring-0 text-blue-900 rounded" 
                        name="remark" id="remark"
                        value={props.value}
                        onChange={props.handleChange}>
                    </textarea>
                </div>
            </div>
            <p className="mt-0.5 text-sm text-gray-500">
                Remark is optional.
            </p>
    </div>
)

export default Remark